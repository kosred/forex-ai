import asyncio
import logging
from pathlib import Path

from ..core.config import Settings
from ..core.system import AutoTuner, HardwareProbe
from ..data.loader import DataLoader
from ..execution.drift_monitor import get_drift_monitor
from ..execution.mt5_state_manager import MT5StateManager
from ..execution.news_service import NewsService
from ..execution.order_execution import OrderExecutor
from ..execution.risk import RiskManager
from ..execution.trade_doctor import TradeDoctor
from ..execution.trading_loop import TradingEngine

# New Services
from ..execution.training_service import TrainingService
from ..features.engine import SignalEngine
from ..features.pipeline import FeatureEngineer
from ..strategy.discovery import AutonomousDiscoveryEngine
from ..training.online_learner import OnlineLearner
from ..training.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class ForexBot:
    """
    The Coordinator.
    Initializes services and delegates execution.
    """

    def __init__(self, settings: Settings):
        self.settings = settings

        # 1. Hardware & Config
        probe = HardwareProbe()
        self.hardware_profile = probe.detect()
        tuner = AutoTuner(settings, self.hardware_profile)
        self.autotune_hints = tuner.apply()

        # 2. Core Components
        self.data_loader = DataLoader(settings)
        self.risk_manager = RiskManager(settings)
        self.feature_engineer = FeatureEngineer(settings)
        self.signal_engine = SignalEngine(settings)
        self.trainer = ModelTrainer(settings)
        self.discovery_engine = AutonomousDiscoveryEngine(Path(settings.system.cache_dir))
        self.drift_monitor = get_drift_monitor(Path(settings.system.cache_dir))
        self.trade_doctor = TradeDoctor(settings)

        # 3. Services
        self.training_service = TrainingService(
            settings, self.data_loader, self.trainer, self.feature_engineer, self.discovery_engine, self.autotune_hints
        )

        self.news_service = NewsService(settings, risk_ledger=getattr(self.risk_manager, "risk_ledger", None))

        # MT5 Manager (lazy init)
        self.mt5_manager = None

        # Online Learner (lazy init)
        self.online_learner = None

    async def train(self, optimize: bool = True, stop_event: asyncio.Event | None = None) -> None:
        """Delegate to TrainingService."""
        await self.training_service.train(optimize, stop_event)

    async def train_global(
        self, symbols: list[str], optimize: bool = True, stop_event: asyncio.Event | None = None
    ) -> None:
        """Delegate to TrainingService."""
        await self.training_service.train_global(symbols, optimize, stop_event)

    async def run(self, paper_mode: bool = True, stop_event: asyncio.Event | None = None) -> None:
        """Initialize runtime components and start the Trading Engine loop."""
        logger.info("Initializing Runtime...")

        try:
            # Connect Data
            await self.data_loader.connect()
            if self.settings.system.mt5_required and not self.data_loader.is_connected():
                raise RuntimeError("MT5 Connection Failed.")

            # Init MT5 State
            self.mt5_manager = MT5StateManager(self.data_loader.mt5_adapter.connection, self.settings)

            # Init Models
            self.signal_engine.load_models("models")
            if not self.signal_engine.models:
                logger.warning("Models missing. Training first...")
                await self.train(optimize=True, stop_event=stop_event)
                # HPC FIX: Post-training memory purge
                import gc
                import torch
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.signal_engine.load_models("models")

            # Init Online Learner
            self.online_learner = OnlineLearner("models")
            self.online_learner.load_models(self.signal_engine.models)

            # Init Executor
            order_executor = OrderExecutor(
                self.settings, self.risk_manager, self.mt5_manager, strategy_ledger=None, risk_ledger=None
            )

            # Create Engine
            engine = TradingEngine(
                settings=self.settings,
                mt5=self.mt5_manager,
                risk=self.risk_manager,
                signal=self.signal_engine,
                doctor=self.trade_doctor,
                news=self.news_service,
                executor=order_executor,
                trainer=self.training_service,
                drift=self.drift_monitor,
                learner=self.online_learner,
            )

            # Run Loop
            await engine.run_loop(stop_event)

        finally:
            # Ensure connection is always closed
            if hasattr(self.data_loader, "disconnect"):
                try:
                    await self.data_loader.disconnect()
                    logger.info("Data loader disconnected.")
                except Exception as e:
                    logger.error(f"Error during data loader disconnect: {e}", exc_info=True)
