import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class StrategyMetrics:
    strategy_id: str
    total_trades: int
    win_rate: float
    profit_factor: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    total_return_pct: float
    avg_win_pct: float
    avg_loss_pct: float
    largest_win_pct: float
    largest_loss_pct: float
    max_drawdown_pct: float
    avg_drawdown_pct: float
    longest_losing_streak: int
    longest_winning_streak: int
    expectancy: float
    kelly_fraction: float
    statistical_significance: float
    monthly_win_rate: float
    positive_months: int
    negative_months: int
    profit_per_trade: float
    avg_trade_duration_hours: float
    trades_per_month: float
    quality_score: float = 0.0
    has_edge: bool = False
    recommendation: str = ""


class StrategyQualityAnalyzer:
    def __init__(self) -> None:
        self.min_sharpe = 1.2
        self.min_profit_factor = 1.5
        self.min_win_rate = 0.50
        self.min_trades = 50
        self.max_dd_acceptable = 0.15
        self.edge_significance_pvalue = 0.05
        logger.info("StrategyQualityAnalyzer initialized")

    def analyze_strategy(
        self, strategy_id: str, trades: pd.DataFrame, initial_balance: float = 100_000
    ) -> StrategyMetrics:
        if trades.empty or len(trades) < 10:
            return self._empty_metrics(strategy_id)

        if "pnl" not in trades.columns:
            logger.warning(f"Strategy {strategy_id}: Missing 'pnl' column")
            return self._empty_metrics(strategy_id)

        if "pnl_pct" not in trades.columns:
            trades["pnl_pct"] = trades["pnl"] / initial_balance

        pnls = trades["pnl"].values
        returns = trades["pnl_pct"].values

        total_trades = len(trades)
        wins = returns[returns > 0]
        losses = returns[returns <= 0]

        win_rate = len(wins) / total_trades if total_trades > 0 else 0.0
        avg_win_pct = np.mean(wins) if len(wins) > 0 else 0.0
        avg_loss_pct = np.mean(losses) if len(losses) > 0 else 0.0  # avg_loss is usually negative in data, keep sign?

        gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
        gross_loss = abs(trades.loc[trades["pnl"] <= 0, "pnl"].sum())
        profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else float("inf")

        equity_curve = initial_balance + trades["pnl"].cumsum()
        running_max = equity_curve.cummax()
        drawdowns = (running_max - equity_curve) / running_max
        max_dd = drawdowns.max()
        avg_dd = drawdowns.mean()

        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)

        total_return = trades["pnl"].sum()
        total_return_pct = total_return / initial_balance
        calmar = (total_return_pct / max_dd) if max_dd > 1e-9 else 0.0

        longest_win_streak = self._longest_streak(trades, win=True)
        longest_loss_streak = self._longest_streak(trades, win=False)

        avg_win_mag = avg_win_pct
        avg_loss_mag = abs(avg_loss_pct)
        expectancy = (win_rate * avg_win_mag) - ((1 - win_rate) * avg_loss_mag)

        kelly = self._calculate_kelly(win_rate, avg_win_mag, avg_loss_mag)

        p_value = self._test_statistical_significance(returns)

        monthly_metrics = self._analyze_monthly_consistency(trades)
        monthly_win_rate = monthly_metrics["win_rate"]

        if "duration" not in trades.columns and "exit_time" in trades.columns and "entry_time" in trades.columns:
            trades["duration"] = (
                pd.to_datetime(trades["exit_time"], utc=True) - pd.to_datetime(trades["entry_time"], utc=True)
            ).dt.total_seconds() / 3600
        avg_duration = trades["duration"].mean() if "duration" in trades.columns else 0.0

        trades_per_month = self._calculate_trade_frequency(trades)

        metrics = StrategyMetrics(
            strategy_id=strategy_id,
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            total_return_pct=total_return_pct,
            avg_win_pct=avg_win_pct,
            avg_loss_pct=avg_loss_pct,  # kept as signed value
            largest_win_pct=np.max(returns) if len(returns) else 0.0,
            largest_loss_pct=np.min(returns) if len(returns) else 0.0,
            max_drawdown_pct=max_dd,
            avg_drawdown_pct=avg_dd,
            longest_losing_streak=longest_loss_streak,
            longest_winning_streak=longest_win_streak,
            expectancy=expectancy,
            kelly_fraction=kelly,
            statistical_significance=p_value,
            monthly_win_rate=monthly_win_rate,
            positive_months=monthly_metrics["positive"],
            negative_months=monthly_metrics["negative"],
            profit_per_trade=np.mean(pnls) if len(pnls) else 0.0,
            avg_trade_duration_hours=avg_duration,
            trades_per_month=trades_per_month,
        )

        self._score_strategy(metrics)
        return metrics

    def _calculate_sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        std_ret = np.std(returns, ddof=1)
        if std_ret < 1e-9:
            return 0.0
        return (mean_ret / std_ret) * np.sqrt(252)

    def _calculate_sortino(self, returns: np.ndarray) -> float:
        if len(returns) < 2:
            return 0.0
        mean_ret = np.mean(returns)
        downside = returns[returns < 0]
        if len(downside) < 2:
            return 0.0
        down_std = np.std(downside, ddof=1)
        if down_std < 1e-9:
            return 0.0
        return (mean_ret / down_std) * np.sqrt(252)

    def _longest_streak(self, trades: pd.DataFrame, win: bool) -> int:
        if win:
            streak_col = (trades["pnl"] > 0).astype(int)
        else:
            streak_col = (trades["pnl"] <= 0).astype(int)

        max_streak = 0
        current_streak = 0
        for val in streak_col:
            if val == 1:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        return max_streak

    def _calculate_kelly(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        if avg_loss < 1e-6 or win_rate >= 1.0 or win_rate <= 0.0:
            return 0.0
        b = avg_win / avg_loss
        p = win_rate
        q = 1 - p
        kelly = (p * b - q) / b
        kelly = max(0.0, min(1.0, kelly))
        return kelly * 0.25  # Fractional Kelly

    def _test_statistical_significance(self, returns: np.ndarray) -> float:
        if len(returns) < 10:
            return 1.0
        t_stat, p_value = stats.ttest_1samp(returns, 0.0, alternative="greater")
        return p_value

    def _analyze_monthly_consistency(self, trades: pd.DataFrame) -> dict[str, Any]:
        if trades.empty or "entry_time" not in trades.columns:
            return {"win_rate": 0.0, "positive": 0, "negative": 0}

        try:
            trades_copy = trades.copy()
            trades_copy["entry_time"] = pd.to_datetime(trades_copy["entry_time"], utc=True)
            trades_copy["month"] = trades_copy["entry_time"].dt.to_period("M")
            monthly_pnl = trades_copy.groupby("month")["pnl"].sum()
            positive = (monthly_pnl > 0).sum()
            negative = (monthly_pnl <= 0).sum()
            total = len(monthly_pnl)
            return {
                "win_rate": positive / total if total > 0 else 0.0,
                "positive": int(positive),
                "negative": int(negative),
            }
        except Exception:
            return {"win_rate": 0.0, "positive": 0, "negative": 0}

    def _calculate_trade_frequency(self, trades: pd.DataFrame) -> float:
        if len(trades) == 0:
            return 0.0
        start = pd.to_datetime(trades["entry_time"].min())
        end = pd.to_datetime(trades["entry_time"].max())
        days = (end - start).days
        months = days / 30.0 if days > 0 else 1.0
        return len(trades) / months

    def _score_strategy(self, metrics: StrategyMetrics) -> None:
        score = 0.0
        if metrics.sharpe_ratio >= 2.0:
            score += 30
        elif metrics.sharpe_ratio >= 1.5:
            score += 23
        elif metrics.sharpe_ratio >= 1.2:
            score += 18
        elif metrics.sharpe_ratio >= 1.0:
            score += 15
        elif metrics.sharpe_ratio >= 0.8:
            score += 10

        if metrics.profit_factor >= 2.5:
            score += 20
        elif metrics.profit_factor >= 2.0:
            score += 15
        elif metrics.profit_factor >= 1.8:
            score += 12
        elif metrics.profit_factor >= 1.5:
            score += 10
        elif metrics.profit_factor >= 1.3:
            score += 5

        if metrics.win_rate >= 0.65:
            score += 15
        elif metrics.win_rate >= 0.60:
            score += 12
        elif metrics.win_rate >= 0.55:
            score += 10
        elif metrics.win_rate >= 0.50:
            score += 8

        if metrics.max_drawdown_pct <= 0.08:
            score += 15
        elif metrics.max_drawdown_pct <= 0.10:
            score += 13
        elif metrics.max_drawdown_pct <= 0.12:
            score += 10
        elif metrics.max_drawdown_pct <= 0.15:
            score += 7

        if metrics.statistical_significance <= 0.01:
            score += 10
        elif metrics.statistical_significance <= 0.05:
            score += 7

        if metrics.monthly_win_rate >= 0.70:
            score += 10
        elif metrics.monthly_win_rate >= 0.60:
            score += 7
        elif metrics.monthly_win_rate >= 0.50:
            score += 5

        metrics.quality_score = min(100.0, score)

        metrics.has_edge = (
            metrics.sharpe_ratio >= self.min_sharpe
            and metrics.profit_factor >= self.min_profit_factor
            and metrics.win_rate >= self.min_win_rate
            and metrics.max_drawdown_pct <= self.max_dd_acceptable
            and metrics.statistical_significance <= self.edge_significance_pvalue
            and metrics.total_trades >= self.min_trades
        )

        if metrics.quality_score >= 80:
            metrics.recommendation = "EXCELLENT"
        elif metrics.quality_score >= 70:
            metrics.recommendation = "GOOD"
        elif metrics.quality_score >= 60:
            metrics.recommendation = "ACCEPTABLE"
        else:
            metrics.recommendation = "POOR"

    def _empty_metrics(self, strategy_id: str) -> StrategyMetrics:
        return StrategyMetrics(
            strategy_id=strategy_id,
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            calmar_ratio=0.0,
            total_return_pct=0.0,
            avg_win_pct=0.0,
            avg_loss_pct=0.0,
            largest_win_pct=0.0,
            largest_loss_pct=0.0,
            max_drawdown_pct=0.0,
            avg_drawdown_pct=0.0,
            longest_losing_streak=0,
            longest_winning_streak=0,
            expectancy=0.0,
            kelly_fraction=0.0,
            statistical_significance=1.0,
            monthly_win_rate=0.0,
            positive_months=0,
            negative_months=0,
            profit_per_trade=0.0,
            avg_trade_duration_hours=0.0,
            trades_per_month=0.0,
        )


class StrategyRanker:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.analyzer = StrategyQualityAnalyzer()
        self.strategy_metrics: dict[str, StrategyMetrics] = {}

    def evaluate_strategies(
        self, strategies: dict[str, pd.DataFrame], initial_balance: float = 100_000
    ) -> list[StrategyMetrics]:
        results = []
        for strategy_id, trades in strategies.items():
            metrics = self.analyzer.analyze_strategy(strategy_id, trades, initial_balance)
            self.strategy_metrics[strategy_id] = metrics
            results.append(metrics)

        results.sort(key=lambda m: m.quality_score, reverse=True)
        self._save_rankings(results)
        return results

    def get_top_strategies(self, n: int = 5, min_quality: float = 60.0) -> list[str]:
        ranked = sorted(self.strategy_metrics.items(), key=lambda x: x[1].quality_score, reverse=True)
        top = [sid for sid, m in ranked[:n] if m.quality_score >= min_quality and m.has_edge]
        return top

    def _save_rankings(self, results: list[StrategyMetrics]) -> None:
        rankings = []
        for m in results:
            rankings.append(
                {
                    "strategy_id": m.strategy_id,
                    "quality_score": m.quality_score,
                    "has_edge": m.has_edge,
                    "recommendation": m.recommendation,
                }
            )
        try:
            with open(self.cache_dir / "strategy_rankings.json", "w") as f:
                json.dump(rankings, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save strategy rankings: {e}")
