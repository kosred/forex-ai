# Project Deep-Dive Findings (forex-ai)

This document summarizes the most important correctness, training, and evaluation issues found in the codebase and what was fixed, plus how the “TA‑Lib / Evo / Genes” components interact with the ML stack.

## 1) Big picture: what the system is doing

At a high level the bot is a *multi-expert ensemble*:

- **FeatureEngineer** builds a large feature matrix (multi-timeframe + market structure + volume/OBI + indicators).
- Multiple **ExpertModel** implementations (trees, deep models, neuroevolution, genetic/TA‑Lib rules, RL agents) each output **3-class probabilities** in a shared convention: `[neutral, buy, sell]`.
- **MetaBlender** stacks those expert probabilities to learn a higher-level “best combination”.
- **SignalEngine** applies gating (confidence/uncertainty) and then **risk + prop constraints** (TradeDoctor/RiskManager) drive sizing and trade allowance.

## 2) What Optuna “value” means (and why it looked “too low”)

Your logs show Optuna trial values like `0.16 … 0.42`. Those numbers are **not PnL**.

Optuna is optimizing an *objective score* computed on a validation split. In this project the optimizer supports two modes:

- **Prop-aware mode (preferred):** uses OHLC metadata to run a fast SL/TP backtest and returns a composite score (profit-like + sharpe-ish + win-rate-ish), while hard-rejecting setups that violate drawdown or don’t trade enough.
- **Accuracy-only fallback:** used if OHLC metadata isn’t available; then the score is just classification accuracy.

If you see values hovering around ~`0.2–0.3`, that often indicates you were in **accuracy-only** mode (a hard 3-class problem) rather than “profit too low”.

## 3) Accuracy ≠ profit (even if labels were built from TP/SL)

Even when labels come from a TP/SL meta-labeling scheme, **accuracy does not uniquely determine profitability**.

Profit depends on:

- **Trade frequency** (accuracy can be inflated by predicting “no trade” too often).
- **Costs** (spread + commission + slippage).
- **Asymmetry** (wins/losses magnitudes, not just win rate).
- **Drawdown constraints** (prop rules can make “good accuracy” unusable).

So: you can have “good accuracy” and still lose money (or violate prop rules), and you can have “mediocre accuracy” but positive expectancy if the errors are small and winners are large.

## 4) Critical fixes applied (profit-impacting correctness, not “strategy changes”)

These fixes were chosen to **avoid changing the unsupervised bot logic** while fixing objective bugs that can invalidate training/backtests:

### A) Fixed label/probability contract across the whole system

- Standardized the action label convention to `{-1, 0, 1}` (sell/neutral/buy) and standardized `predict_proba()` ordering to `[neutral, buy, sell]`.
- Updated tree/deep/ONNX paths + stacking so they all agree.

This removes silent “wrong-way” trades caused by class-order mismatches.

### B) Fixed XGBoost training crash

You hit: `Expected: [0 1 2], got [-1  0  1]`.

XGBoost requires contiguous class indices; the training path now encodes labels for XGBoost internally and decodes/reorders probabilities back to the project convention.

### C) Fixed multi-timeframe lookahead leakage

Higher-timeframe features were forward-filled onto the base timeframe without shifting to the **last closed** HTF candle.

This is a classic leak that can make both “accuracy” and backtests look unrealistically good.

HTF features are now shifted by 1 candle before merging.

### D) Fixed live “repainting candle” usage in MT5 loader

Live fetching included the *current forming candle* which is not stable and can repaint.

Loader now pulls from `start_pos=1` so the newest bar used is the last **completed** candle.

### E) Fixed global training metadata so Optuna can optimize profit-aware objective

Global pooled training previously dropped OHLC metadata, forcing Optuna into accuracy-only scoring even when OHLC existed.

Now global training pools **symbol-aware OHLC metadata** for the optimizer, and the optimizer aggregates profit score per symbol (to avoid multi-symbol mixing).

## 5) RLlib: why `pip install -U "ray[all]"` / `ray[rllib]` doesn’t install

On **Windows + Python 3.13**, pip currently cannot find a compatible `ray` wheel, so `ray[rllib]` / `ray[all]` fails with “no matching distribution”.

Practical options:

- Use **Python 3.11 or 3.12** on Windows.
- Or run the bot on **WSL/Ubuntu** (or Linux server) with a supported Python.

Also note: the current RLlib wrapper in this repo is a **classification reward** (`+1/-1` for correct class), not PnL reward.

## 5b) "Warning-free" runs (what was fixed vs what must be configured)

To keep runtime output clean without hiding real problems:

- **Optuna ExperimentalWarning removed** by not using experimental `TPESampler(multivariate=..., group=...)` flags.
- **TabNet DeprecationWarning fixed at the source** by auto-patching `pytorch_tabnet/multiclass_utils.py` during dependency bootstrap to import `spmatrix` from `scipy.sparse` (SciPy 2.0 compatible).
- **Dependency bootstrap hardened** (Windows): cleans pip `~...` leftovers (fixes `Ignoring invalid distribution ~umpy`) and re-execs once after installs/upgrades (fixes in-process NumPy/Pandas DLL reload crashes).
- **ONNX CPU inference actually works now**: fixed ONNX inference loader; ONNX auto-enables when `models/onnx/export_manifest.joblib` exists.
- **ONNX export is optional** (Settings default `models.export_onnx: false`). When enabled (e.g., in `config.yaml`), exports the supported non-PyTorch models once for fast CPU inference (`lightgbm`, `random_forest`, `elasticnet`, `catboost`).
- **Torch->ONNX export is disabled by default** to avoid noisy failures (shape requirements + `onnxscript`). Enable only if needed via `FOREX_BOT_EXPORT_TORCH_ONNX=1`.
- **Metadata-required experts** (`genetic`, `rl_ppo`, `rl_sac`) are skipped automatically when OHLC metadata is unavailable (e.g., pooled multi-symbol training where metadata is intentionally disabled to prevent leakage).

## 6) How TA‑Lib / Genes / Evo increase performance & consistency in this architecture

### TA‑Lib “Genes” (rule strategy discovery)

- `GeneticStrategyExpert` evolves combinations of TA‑Lib indicators into rule strategies using OHLC metadata.
- It can also “bridge” discovered strategies saved in `cache/talib_knowledge.json`.
- Output is converted into `[neutral, buy, sell]` probabilities so it can be ensembled with ML experts.

### Neuroevolution (fast learner, different failure modes)

- `EvoExpertCMA` uses evolutionary optimization to fit a small model (different inductive bias than trees/deep nets).
- It’s useful because it often fails differently than gradient-trained models, which is valuable in an ensemble.

### Ensemble stacking (how it all becomes one decision)

- Each expert produces probabilities.
- `MetaBlender` learns how to weight experts based on holdout performance.
- `SignalEngine` can further weight experts using fast-backtest metrics (profit/drawdown) when available.

This is how “genes/evo” become directly usable by the trading decision without dumping thousands of extra raw TA features into every ML model.

## 7) Clearing caches/artifacts (models/features/optuna journals)

Two ways:

1) Run the cleanup script directly:
   - `python clean_artifacts.py --cache`
   - `python clean_artifacts.py --models`
   - `python clean_artifacts.py --all`

2) Use the new CLI flags:
   - `python forex-ai.py --clean cache`
   - `python forex-ai.py --clean models`
   - `python forex-ai.py --clean all`
   - Add `--clean-only` to exit after cleaning.

## 8) What to expect for profit

No code change can honestly promise “4% per month” — the realistic expectation is:

- After removing leakage and label/proba mismatches, **backtests and validation scores typically get less optimistic but more real**.
- If the previous system benefited from leakage, you may see lower “accuracy” but improved live stability.
- Profit ultimately depends on your broker conditions, execution, and how well the risk layer enforces prop constraints during live regimes you didn’t train on.

If you want, the next step is to run a walk-forward / out-of-sample evaluation and report:

- `net_profit`, `max_dd`, `profit_factor`, `win_rate`, `trades`, and “negative months count”.

## 9) MT5 (Windows): IPC timeout and terminal path

If you see `(-10005, 'IPC timeout')`, it is almost always environmental (terminal not running / wrong user session / wrong path).

Fixes in code:

- Auto-detects `terminal64.exe` under `Program Files` when `system.mt5_terminal_path` is empty.
- On IPC timeout, **launches MT5 once** and retries initialization.

What you still must do:

- Ensure MT5 is installed at `C:\\Program Files\\MetaTrader 5\\terminal64.exe` (or set `system.mt5_terminal_path` explicitly).
- Ensure MT5 is running, logged in, and Python + MT5 run under the **same Windows user** and **same privilege level** (both admin or both non-admin).
