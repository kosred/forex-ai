import logging
import os
from pathlib import Path

import torch
import yaml

from .config import Settings

logger = logging.getLogger(__name__)


class AutoConfigurator:
    """
    Automatically tunes the bot configuration based on:
    1. Hardware (GPU vs CPU)
    2. Account Balance (Prop Firm detection)
    3. Available Data
    4. User intent (Zero-config)
    """

    def __init__(self, settings: Settings, config_path: str = "config.yaml"):
        self.settings = settings
        self.config_path = Path(config_path)

    def auto_tune(self, force: bool = False) -> Settings:
        """
        Run the auto-tuning process.
        If config.yaml doesn't exist or force=True, we overwrite/update settings.
        """
        if self.config_path.exists() and not force:
            # Even if config exists, we might want to soft-tune hardware if it's set to "auto"
            self._tune_hardware()
            return self.settings

        logger.info("Auto-configuring bot for optimal performance...")

        # 1. Hardware Tuning
        self._tune_hardware()

        # 2. Risk / Prop Firm Tuning (Heuristic based on default balance)
        # Note: In a real scenario, we'd fetch balance from MT5 first, but we need config to connect to MT5.
        # So we make a best guess or default to 'Prop Firm' safe mode.
        self._tune_risk_profile()

        # 3. Model Selection
        self._tune_model_selection()

        # 4. Save to config.yaml so the user can see what happened
        self._save_config()

        return self.settings

    def _tune_hardware(self):
        """
        Advanced Hardware Auto-Tuning (Hybrid Parallelism).
        Detects Single GPU, Multi-GPU (DDP), or Massive CPU Cluster.
        """
        if self.settings.system.enable_gpu_preference == "cpu":
            self._tune_cpu_only()
            return

        try:
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                cpu_count = os.cpu_count() or 1

                logger.info(f"Hardware Detected: {gpu_count}x {gpu_name} ({vram_gb:.1f} GB VRAM), {cpu_count} CPU Cores")

                self.settings.system.enable_gpu = True
                self.settings.system.device = "cuda"
                self.settings.system.num_gpus = gpu_count

                # --- 1. Multi-GPU Cluster (DDP) ---
                if gpu_count > 1:
                    logger.info(f"?? Multi-GPU Cluster ({gpu_count} GPUs). Enabling Hybrid DDP Strategy.")
                    # Deep Learning gets DDP (All GPUs work on one model)
                    self.settings.models.enable_transformer_expert = True
                    self.settings.models.use_rl_agent = True
                    # Scale batch size roughly: 128 base * VRAM factor * GPU count?
                    # Actually, for DDP, effective batch size is local_batch * world_size.
                    # We keep local batch high to saturate tensor cores.
                    if vram_gb > 40:  # A6000/A100
                        self.settings.models.train_batch_size = 4096
                    elif vram_gb > 20:  # 3090/4090
                        self.settings.models.train_batch_size = 2048
                    else:
                        self.settings.models.train_batch_size = 512
                    
                    # Tree models (XGBoost/CatBoost) should run on CPU in parallel to GPUs?
                    # Or we use 'Task Parallelism' for them.
                    # If we have massive CPUs, let's use them.
                    self.settings.system.n_jobs = max(1, cpu_count // 2) 

                # --- 2. Single High-End GPU (A6000/4090) ---
                elif vram_gb > 20:
                    logger.info("?? High-End Single GPU. Maximizing Batch Size.")
                    self.settings.models.train_batch_size = 4096 if vram_gb > 40 else 1024
                    self.settings.models.enable_transformer_expert = True
                    self.settings.models.use_rl_agent = True
                    # Leave CPUs for data loading / ensemble
                    self.settings.system.n_jobs = max(1, cpu_count - 4)

                # --- 3. Standard GPU (T4/3060) ---
                else:
                    logger.info("Standard GPU. Using balanced settings.")
                    self.settings.models.train_batch_size = 128
                    self.settings.models.enable_transformer_expert = True
                    self.settings.system.n_jobs = max(1, cpu_count - 2)

            else:
                self._tune_cpu_only()

        except Exception as e:
            logger.warning(f"Hardware tuning failed: {e}")
            self._tune_cpu_only()

    def _tune_cpu_only(self):
        """Fallback for CPU-only environments (including massive CPU nodes)."""
        cpu_count = os.cpu_count() or 1
        logger.info(f"No GPU detected. optimizing for {cpu_count} CPU Cores.")
        self.settings.system.enable_gpu = False
        self.settings.system.device = "cpu"
        self.settings.models.train_batch_size = 64
        
        # Massive CPU Node (e.g. 250 cores)
        if cpu_count > 64:
            logger.info("?? Massive CPU Cluster detected. Enabling max parallelism for Tree Models.")
            self.settings.system.n_jobs = cpu_count  # Use everything
            # We can still train Transformers on CPU if we have 250 cores, it's just slow per-epoch
            self.settings.models.enable_transformer_expert = True 
        else:
            self.settings.system.n_jobs = max(1, cpu_count - 1)

    def _tune_risk_profile(self):
        """
        Detect if we look like a prop firm account and apply strict rules.
        """
        # Default to strict Prop Firm rules for "Safety First"
        self.settings.risk.prop_firm_rules = True
        self.settings.risk.consistency_tracking = True
        self.settings.risk.daily_drawdown_limit = 0.04  # 4% Hard limit (safer than 5%)
        self.settings.risk.total_drawdown_limit = 0.08  # 8% Hard limit
        self.settings.risk.max_risk_per_trade = 0.01  # 1% Max risk
        self.settings.risk.risk_per_trade = 0.005  # 0.5% Base risk
        self.settings.risk.recovery_mode_enabled = True

        logger.info("Risk profile: prop firm safe mode (Max DD 4%, Risk 0.5%)")

    def _tune_model_selection(self):
        """Select best robust models."""
        # Enable all robust tree models
        if "catboost" not in self.settings.models.ml_models:
            self.settings.models.ml_models.append("catboost")

        # Enable Online Learning features
        self.settings.news.enable_news = True
        self.settings.news.news_trade_confidence_threshold = 0.85  # High confidence for news

    def _save_config(self):
        """Persist the tuned settings to config.yaml."""
        try:
            # Convert Pydantic settings to dict, respecting nested models
            data = self.settings.model_dump()

            # Simple header
            with open(self.config_path, "w") as f:
                f.write("# Auto-Generated Configuration by ForexBot AI\n")
                f.write(f"# Date: {logging.Formatter().formatTime(logging.LogRecord('', 0, '', 0, '', '', None))}\n")
                yaml.dump(data, f, default_flow_style=False)

            logger.info(f"💾 Saved optimized configuration to {self.config_path}")
        except Exception as e:
            logger.warning(f"Failed to save auto-config: {e}")
