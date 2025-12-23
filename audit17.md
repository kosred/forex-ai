# Final Consolidated Production Audit (Post Deep-Dive)

**Date:** 2025-12-21
**Evaluator:** Gemini CLI Agent
**Scope:** 113 Python Files + System Architecture
**Summary:** The bot has a high "research" quality but contains several **Production Blockers** that will cause failure in live environments or during heavy historical training.

---

## 🛑 1. Critical "Stop the Line" Bugs (Logical & Runtime)

### **A. The "Death Penalty" Objective Function**
*   **File:** `src/forex_bot/training/optimization.py` (Line ~550)
*   **Defect:** Trials with < 30 trades or high drawdown return a hard `-1e9` score.
*   **Reason for KAN/TiDE Failure:** Deep models take longer to build confidence. Early trials often produce "Neutral" signals (0 trades), triggering the death penalty. Optuna then prunes these configurations as "Bad," preventing the models from ever finding the right parameters.
*   **Fix:** Reduce `prop_min_trades` to 5 for Optuna trials and use a continuous penalty instead of a hard floor.

### **B. KAN GPU Numerical Divergence**
*   **File:** `src/forex_bot/models/kan_gpu.py`
*   **Defect:** Missing `StandardScaler` (Mean/Std) logic.
*   **Impact:** KANs use B-Splines defined on a fixed grid (typically $[-1, 1]$). Passing raw RSI (0-100) or ATR data forces evaluation far outside the spline boundaries, causing `NaN` gradients and total model failure.
*   **Fix:** Copy the robust normalization logic from `kan.py` (CPU version) into the GPU version.

### **C. Corrupted Validation (Lookahead Bias)**
*   **Files:** `src/forex_bot/training/cpcv.py` (Line 315) and `src/forex_bot/training/evaluation.py` (Line 75).
*   **Defect:** `future = np.roll(close, -1)`.
*   **Impact:** This is "Time Travel." The model is shown tomorrow's price during the validation step. All CPCV and "Quick Backtest" results are artificially inflated and functionally useless for predicting live performance.
*   **Fix:** Replace `np.roll` with proper point-in-time return calculation using the *next* closed candle.

### **D. Fatal Async Nesting Crash**
*   **File:** `src/forex_bot/execution/mt5_state_manager.py` (Line 520)
*   **Defect:** `asyncio.run(_async_write())` called from within a running loop.
*   **Impact:** Python will raise a `RuntimeError` and crash the entire bot the moment it attempts to save an entry feature vector during live trading.
*   **Fix:** Replace with `asyncio.create_task(_async_write())`.

---

## 🐢 2. Performance & Scaling Bottlenecks

### **A. $O(N^2)$ News Feature Builder**
*   **File:** `src/forex_bot/data/news/client.py` (Line 695)
*   **Defect:** Python `for` loop using `.loc` slicing for every bar in the dataset.
*   **Impact:** Processing 5 years of M1 data will take hours and likely hang the system.
*   **Fix:** Use `pandas.merge_asof` to align news events to price bars in one vectorized operation.

### **B. Blocking Network Search**
*   **File:** `src/forex_bot/data/news/searchers.py` (Line 33)
*   **Defect:** Synchronous `requests.post` inside the async trading loop.
*   **Impact:** The bot will "freeze" for 2-30 seconds every time it fetches headlines, causing it to miss price ticks and execution windows.
*   **Fix:** Migrate to `httpx.AsyncClient`.

---

## 🏗️ 3. Architectural & Deployment Risks

### **A. "Zombie" Evolution Pool**
*   **File:** `src/forex_bot/strategy/genetic.py` (Line 130)
*   **Defect:** Elite strategies are never re-evaluated (`if fitness == 0.0`).
*   **Impact:** Strategies that were "lucky" 6 months ago will stay in the pool forever, even if the market regime has changed and they are now losing money.
*   **Fix:** Re-run the gauntlet on the entire population every generation.

### **B. RLlib "Amnesia" (Persistence Stubs)**
*   **File:** `src/forex_bot/models/rllib_agent.py` (Lines 135-140)
*   **Defect:** `save` and `load` methods are empty.
*   **Impact:** RLlib agents (PPO/SAC) lose 100% of their training progress every time the bot restarts.
*   **Fix:** Implement `self.algo.save()` and `self.algo.restore()`.

### **C. Broken Weighted Voting**
*   **File:** `src/forex_bot/features/indicators.py` (Line 315)
*   **Defect:** `weighted_vote` logic is actually a simple mean.
*   **Impact:** The genetic optimizer's efforts to tune indicator weights are ignored by the mixer.
*   **Fix:** Multi-multiply signal vectors by the weight dict before summing.

---

## **Final Production Roadmap**

| Priority | Action | File |
| :--- | :--- | :--- |
| **CRITICAL** | Remove `np.roll` lookahead | `cpcv.py`, `evaluation.py` |
| **CRITICAL** | Add scaler to GPU models | `kan_gpu.py`, `tabnet_gpu.py` |
| **CRITICAL** | Fix nested `asyncio.run` | `mt5_state_manager.py` |
| **HIGH** | Lower Optuna trade floor | `optimization.py` |
| **HIGH** | Vectorize news proximity | `news/client.py` |
| **MED** | Fix Weighted Voting | `indicators.py` |
| **MED** | Implement RL saving | `rllib_agent.py` |

**Audit Signed Off. Ready for Remediation.**
