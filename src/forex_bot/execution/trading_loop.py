import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ..core.config import Settings
from ..execution.drift_monitor import ConceptDriftMonitor
from ..execution.meta_controller import PropMetaState
from ..execution.mt5_state_manager import MT5StateManager
from ..execution.news_service import NewsService
from ..execution.order_execution import OrderExecutor
from ..execution.risk import RiskManager
from ..execution.trade_doctor import TradeDoctor

# Note: ForexBot import removed to avoid circular dependency
# TradingEngine receives individual components, not the bot itself
from ..execution.training_service import TrainingService
from ..features.engine import SignalEngine
from ..models.unsupervised import MarketRegimeClassifier
from ..training.online_learner import OnlineLearner

logger = logging.getLogger(__name__)


class TradingEngine:
    """
    The main event loop for live trading.
    Decoupled from initialization and training logic.
    """

    def __init__(
        self,
        settings: Settings,
        mt5: MT5StateManager,
        risk: RiskManager,
        signal: SignalEngine,
        doctor: TradeDoctor,
        news: NewsService,
        executor: OrderExecutor,
        trainer: TrainingService,
        drift: ConceptDriftMonitor,
        learner: OnlineLearner | None,
    ):
        self.settings = settings
        self.mt5 = mt5
        self.risk = risk
        self.signal = signal
        self.doctor = doctor
        self.news = news
        self.executor = executor
        self.trainer = trainer
        self.drift = drift
        self.learner = learner

        self._poll_interval = settings.system.poll_interval_seconds
        self._consecutive_failures = 0

        # State for drift monitoring (requires previous prediction vs current outcome)
        self.last_prob = None
        self.last_close_price = None
        self.last_signal = None

        # Unsupervised Regime Classifier
        self.regime_classifier = MarketRegimeClassifier()
        model_path = Path("models")
        try:
            loaded = MarketRegimeClassifier.load(model_path)
            if loaded:
                self.regime_classifier = loaded
                logger.info(f"Loaded Regime Classifier: {self.regime_classifier.regime_map}")
        except Exception as e:
            logger.warning(f"Failed to load regime classifier: {e}")

        self.iters = 0
        self._market_closed_logged = False

    async def run_loop(self, stop_event: asyncio.Event | None = None) -> None:
        """Main infinite loop."""
        logger.info("Starting Trading Engine Loop...")

        while True:
            if stop_event and stop_event.is_set():
                logger.info("Stop requested.")
                break

            try:
                self.iters += 1

                # 1. Market Hours Check
                if not self.risk.is_trading_session():
                    now = datetime.now(UTC)
                    weekday = now.weekday()  # 0=Mon, 6=Sun

                    # Weekend Logic (Saturday=5, Sunday=6)
                    if weekday >= 5:
                        if not self._market_closed_logged:
                            logger.info("Market closed (Weekend). Sleeping intelligently until Monday...")
                            self._market_closed_logged = True
                        # Sleep for 1 hour on weekends to save resources
                        await asyncio.sleep(3600)
                        continue
                    else:
                        if not self._market_closed_logged:
                            logger.info("Market closed (Session break). Sleeping...")
                            self._market_closed_logged = True
                        await asyncio.sleep(60)
                        continue
                else:
                    # Reset logging flag when market opens
                    if self._market_closed_logged:
                        logger.info("Market Open! Resuming trading...")
                        self._market_closed_logged = False

                # 2. Data Fetch (REORDERED: Fetch first to have frames for Doctor)
                frames = await self.trainer.data_loader.get_live_data(self.settings.system.symbol)
                if not frames:
                    self._consecutive_failures += 1
                    if self._consecutive_failures > 10:
                        logger.error("Critical data failure.")
                        await asyncio.sleep(60)
                    await asyncio.sleep(5)
                    continue
                self._consecutive_failures = 0

                # Get current close for drift calculation
                current_close = None
                df_m1 = None
                try:
                    df_m1 = frames.get(self.settings.system.base_timeframe)
                    if df_m1 is None:
                        df_m1 = frames.get("M1")
                    if df_m1 is not None and not df_m1.empty:
                        current_close = float(df_m1.iloc[-1]["close"])
                except Exception:
                    pass

                # 3. Regime Update (Thinking Step)
                regime = "Normal"
                if df_m1 is not None and not df_m1.empty:
                    # Periodically refit (Self-Tuning)
                    if self.iters % 100 == 0:
                        try:
                            # Use daily data if available for broader context, else M1
                            fit_df = frames.get("D1", df_m1)
                            self.regime_classifier.fit(fit_df)
                            self.regime_classifier.save(Path("models"))
                        except Exception as e:
                            logger.warning(f"Regime refit failed: {e}")

                    # Predict current state
                    regime = self.regime_classifier.predict(df_m1)

                # 4. MT5 Sync & Doctor (Now Doctor receives frames)
                await self.mt5.sync_with_mt5()
                positions = list(self.mt5.cached_positions)

                # Run Trade Doctor with frames
                diagnoses = self.doctor.diagnose(positions, frames)

                # Execute close instructions from Doctor
                for instruction in diagnoses:
                    try:
                        await self.executor.close_position(
                            instruction.ticket, instruction.volume, f"Doctor: {instruction.reason}"
                        )
                        logger.info(f"Doctor closed #{instruction.ticket}: {instruction.reason}")
                    except Exception as e:
                        logger.warning(f"Failed to execute doctor instruction for #{instruction.ticket}: {e}")

                # 5. News & Policy
                news_policy = await self.news.get_news_policy(self.settings.system.symbol)
                self.risk.update_news_state(news_policy)

                # 6. Feature Engineering & Signal
                news_feats = self.news.get_news_features(frames[self.settings.system.base_timeframe].index)
                dataset = self.trainer.feature_engineer.prepare(
                    frames,
                    news_features=news_feats,
                    symbol=self.settings.system.symbol,
                )

                # 7. Signal
                result = self.signal.generate_ensemble_signals(dataset)

                # 8. Drift Check (Corrected Logic: Compare LAST prediction with CURRENT outcome)
                if (
                    self.drift
                    and self.last_prob is not None
                    and self.last_close_price is not None
                    and current_close is not None
                    and self.last_signal is not None
                    and int(self.last_signal) != 0
                ):
                    # Determine true label for the *previous* period
                    # If price went up, y_true=1 (Buy). Down, y_true=-1 (Sell). Flat, y_true=0 (Neutral).
                    price_change = current_close - self.last_close_price
                    if price_change > 1e-5:
                        y_true = 1  # Buy/Up
                    elif price_change < -1e-5:
                        y_true = -1  # Sell/Down
                    else:
                        y_true = 0  # Neutral

                    # Update monitor with (Actual, Predicted_Prob_From_Last_Step)
                    self.drift.update(y_true, self.last_prob)

                    if self.drift.should_retrain():
                        logger.warning("Concept drift detected; triggering background retraining...")
                        task = asyncio.create_task(self._trigger_retraining())
                        if not hasattr(self, "_background_tasks"):
                            self._background_tasks = set()
                        self._background_tasks.add(task)
                        task.add_done_callback(self._background_tasks.discard)

                # Store current state for NEXT iteration's drift check
                if result and hasattr(result, "probs"):
                    self.last_prob = result.probs
                if result and hasattr(result, "signal"):
                    self.last_signal = result.signal
                if current_close is not None:
                    self.last_close_price = current_close

                # 9. Execution (With Thinking Brain)
                # We need to manually invoke risk check to inject the regime
                equity = self.mt5.get_real_equity()

                # Override internal risk check to inject regime
                meta_state = PropMetaState(
                    daily_dd_pct=(self.risk.day_start_equity - equity) / self.risk.day_start_equity
                    if self.risk.day_start_equity > 0
                    else 0.0,
                    volatility_regime=self._infer_vol_regime(regime),
                    recent_win_rate=sum(self.risk.rolling_outcomes) / len(self.risk.rolling_outcomes)
                    if self.risk.rolling_outcomes
                    else 0.5,
                    consecutive_losses=self.risk.consecutive_losses,
                    model_confidence=result.confidence,
                    hour_of_day=datetime.now(UTC).hour,
                    market_regime=regime,
                )

                # Get smart parameters
                risk_mult, req_conf, allowed = self.risk.meta_controller.get_risk_parameters(meta_state)

                # Inject back into risk manager for this tick
                # We can't easily monkey-patch, but we can respect the outcome
                if not allowed:
                    logger.info(f"Meta-Controller blocked trade (Regime: {regime})")
                elif result.signal != 0 and result.confidence >= req_conf:
                    # Check repeat mistake guard
                    if self.learner and self.learner.is_repeat_mistake(dataset.X):
                        logger.warning("Similarity guard blocked trade (matches past loss pattern)")
                    else:
                        # Pass risk_mult implicitly by scaling size or handling inside executor?
                        # Executor calls risk_manager.calculate_position_size which calls meta_controller.
                        # BUT risk_manager.calculate_position_size re-creates PropMetaState without our regime!
                        # FIX: We need to patch risk manager or pass regime to it.
                        # For now, we trust risk_manager to re-calculate, but we need to inject regime into it?
                        # No, risk_manager doesn't have 'market_regime' field yet.
                        # Wait, we updated PropMetaState definition globally in meta_controller.py
                        # But risk.py creates PropMetaState. It needs to know about 'market_regime'.
                        # See next step.

                        await self.executor.execute_signal(
                            result,
                            self.mt5.get_real_equity(),
                            frames,
                            advice_stance=self.news.last_advice.get("stance") if self.news.last_advice else None,
                        )

                # 10. Online Learning Update
                if self.learner:
                    # Update with recent closed positions (wins/losses)
                    closed_positions = await self.mt5.get_recently_closed_positions(limit=10)
                    for closed_pos in closed_positions:
                        if not hasattr(closed_pos, "_processed_for_learning"):
                            # Extract entry features if available
                            entry_features = self.mt5.entry_feature_store.get(closed_pos.ticket)
                            if entry_features:
                                profit = closed_pos.profit
                                label = 1 if profit > 0 else -1 if profit < 0 else 0
                                weight = abs(profit) / max(closed_pos.volume * 100, 1.0)  # R-multiple proxy

                                self.learner.add_sample(
                                    pd.DataFrame([entry_features]), pd.Series([label]), weight=weight
                                )
                                closed_pos._processed_for_learning = True

                await asyncio.sleep(self._poll_interval)

            except Exception as e:
                logger.error(f"Loop error: {e}", exc_info=True)
                await asyncio.sleep(5)

    def _infer_vol_regime(self, regime_str: str) -> str:
        if "Volatile" in regime_str:
            return "high"
        if "Quiet" in regime_str:
            return "low"
        return "normal"

    async def _trigger_retraining(self) -> None:
        """Background retraining triggered by drift detection."""
        try:
            logger.info("Starting drift-triggered retraining...")
            await self.trainer.train_incremental_all()
            logger.info("Drift retraining completed")
            self.drift.reset_after_retrain()
        except Exception as e:
            logger.error(f"Drift retraining failed: {e}", exc_info=True)
