from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

from ..models.base import ExpertModel
from ..models.trees import LightGBMExpert

logger = logging.getLogger(__name__)


class OnlineLearner:
    """
    Handles incremental learning (refitting) of models based on real-time trade outcomes.
    Also implements 'Novelty/History Guard' to prevent repeating past mistakes.
    """

    def __init__(
        self,
        models_dir: str,
        buffer_size: int = 1000,
        min_samples_for_update: int = 50,
        update_frequency: int = 10,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.buffer_size = buffer_size
        self.min_samples_for_update = min_samples_for_update
        self.update_frequency = update_frequency  # Update every N samples added

        self.buffer: deque = deque(maxlen=buffer_size)
        self.samples_since_update = 0

        self.loss_archive: list[np.ndarray] = []
        self.loss_archive_limit = 5000

        self.models: dict[str, ExpertModel] = {}
        self.scaler = None

    def load_models(self, models_dict: dict[str, ExpertModel]) -> None:
        self.models = models_dict
        bundle_path = self.models_dir / "models.joblib"
        try:
            bundle = joblib.load(bundle_path)
            if isinstance(bundle, dict):
                self.scaler = bundle.get("scaler")
            else:
                self.scaler = None
        except FileNotFoundError:
            self.scaler = None
        except EOFError as exc:
            # Corrupt/partial file (common when interrupted while writing). Repair best-effort and continue.
            self.scaler = None
            logger.info(f"models.joblib is unreadable ({exc}); repairing thin bundle and continuing.")
            self._repair_models_bundle()
        except Exception as exc:
            self.scaler = None
            logger.info(f"Online learner could not load models bundle ({bundle_path}): {exc}")

    def _repair_models_bundle(self) -> None:
        """
        Best-effort repair of `models.joblib` to avoid repeated EOFErrors at startup.
        """
        bundle_path = self.models_dir / "models.joblib"
        payload = {"models": {}, "model_names": list(self.models.keys()), "scaler": None, "schema_version": 2}
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            tmp = bundle_path.with_name(bundle_path.name + ".tmp")
            joblib.dump(payload, tmp)
            tmp.replace(bundle_path)
        except Exception:
            return

    def add_sample(self, x: pd.DataFrame, y: pd.Series, weight: float = 1.0) -> None:
        """
        Add a completed trade result to the learning buffer.
        y: 1 (Win), -1 (Loss), 0 (Neutral)
        """
        if y.iloc[0] < 0:  # Loss
            try:
                vec = x.iloc[0].to_numpy(dtype=np.float32)
                self._add_to_loss_archive(vec)
            except Exception as e:
                logger.warning(f"Failed to archive loss vector: {e}")

        self.buffer.append((x, y, weight))
        self.samples_since_update += 1

        if self.samples_since_update >= self.update_frequency and len(self.buffer) >= self.min_samples_for_update:
            self.update_models()
            self.samples_since_update = 0

    def _add_to_loss_archive(self, vector: np.ndarray) -> None:
        """Store bad decision vectors to avoid repeating them."""
        if len(self.loss_archive) >= self.loss_archive_limit:
            self.loss_archive.pop(0)
        self.loss_archive.append(vector)

    def is_repeat_mistake(self, current_features: pd.DataFrame, threshold: float = 0.95) -> bool:
        """
        Check if current market state matches a past disaster.
        Returns True if this looks like a 'Bad Idea'.
        """
        if not self.loss_archive:
            return False

        try:
            curr_vec = current_features.iloc[0].to_numpy(dtype=np.float32).reshape(1, -1)
            archive_matrix = np.stack(self.loss_archive)

            sims = cosine_similarity(curr_vec, archive_matrix)
            max_sim = np.max(sims)

            if max_sim > threshold:
                logger.warning(f"Similarity guard: current setup matches a past loss ({max_sim:.2f}). Blocking.")
                return True
        except Exception as e:
            logger.warning(f"Similarity check failed: {e}")

        return False

    def update_models(self) -> bool:
        """
        Perform incremental training (refitting) on all supported models.
        """
        if len(self.buffer) < self.min_samples_for_update:
            return False

        logger.info(f"🔄 Online Learning: Updating models with {len(self.buffer)} recent trades...")

        x_list, y_list, w_list = zip(*self.buffer, strict=False)
        try:
            x_batch = pd.concat(x_list, axis=0)
            y_batch = pd.concat(y_list, axis=0)
            x_batch = x_batch.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        except Exception as e:
            logger.error(f"Online batch preparation failed: {e}")
            return False

        updated_count = 0

        for name, model in self.models.items():
            try:
                if isinstance(model, LightGBMExpert):
                    if hasattr(model.model, "booster_"):
                        model.model.fit(
                            x_batch,
                            y_batch,
                            init_model=model.model.booster_,
                            verbose=False,
                        )
                        updated_count += 1

                elif hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
                    self._update_pytorch_model(model, x_batch, y_batch)
                    updated_count += 1

            except Exception as e:
                logger.warning(f"Failed to update {name}: {e}")

        logger.info(f"Successfully updated {updated_count} models.")
        return True

    def _update_pytorch_model(self, expert: ExpertModel, x: pd.DataFrame, y: pd.Series) -> None:
        """Standard SGD step for PyTorch models."""
        model = expert.model
        device = getattr(expert, "device", "cpu")

        model.train()
        x_tens = torch.as_tensor(x.values, dtype=torch.float32, device=device)

        y_mapped = y.values + 1
        y_tens = torch.as_tensor(y_mapped, dtype=torch.long, device=device)

        criterion = torch.nn.CrossEntropyLoss()

        # Persist optimizer state to maintain momentum
        if not hasattr(expert, "_online_optimizer"):
            expert._online_optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

        optimizer = expert._online_optimizer

        optimizer.zero_grad(set_to_none=True)
        try:
            use_amp = torch.cuda.is_available() and device != "cpu" and hasattr(torch, "amp")
            if use_amp:
                with torch.amp.autocast("cuda", enabled=True):
                    out = model(x_tens)
                    loss = criterion(out, y_tens)
            else:
                out = model(x_tens)
                loss = criterion(out, y_tens)

            loss.backward()
            optimizer.step()
        except Exception as e:
            logger.warning(f"PyTorch update step failed: {e}")

    def snapshot(self) -> dict[str, Any]:
        """Return serializable state."""
        snap_buffer = []
        for i in range(min(len(self.buffer), 100)):
            snap_buffer.append(self.buffer[i])

        return {
            "loss_archive": [v.tolist() for v in self.loss_archive],
            "buffer_sample_count": len(self.buffer),
        }

    def restore(self, state: dict[str, Any]) -> None:
        """Restore state from snapshot."""
        try:
            if "loss_archive" in state:
                self.loss_archive = [np.array(v) for v in state["loss_archive"]]
        except Exception as e:
            logger.warning(f"Online learning update failed: {e}", exc_info=True)
