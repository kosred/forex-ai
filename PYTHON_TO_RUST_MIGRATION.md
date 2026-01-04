# Python -> Rust Migration Map (forex-ai)

Date: 2026-01-04

## Scope
- Python deps from `requirements-hpc.txt`
- Model registry from `src/forex_bot/models/registry.py`

Legend:
- Verified = linked source
- Candidate = exists but early/WIP/limited
- Pending = not researched yet

## Verified mappings (with sources)

### Data & Processing
| Python | Rust | Maturity | Notes / Source |
| --- | --- | --- | --- |
| pandas | `polars` | Stable | Polars DataFrame library (docs). https://docs.pola.rs/docs/rust/dev/polars/ |
| numpy | `ndarray` | Stable | N-dimensional arrays. https://github.com/rust-ndarray/ndarray |
| pyarrow / parquet | `arrow` / `parquet` | Stable | Arrow Rust + Parquet crates. https://github.com/apache/arrow-rs / https://arrow.apache.org/rust/parquet/index.html |
| TA-Lib | `ta-lib-in-rust` | Stable | Rust TA-Lib implementation. https://lib.rs/crates/ta-lib-in-rust |

### Deep Learning & Time-Series
| Python | Rust | Maturity | Notes / Source |
| --- | --- | --- | --- |
| PyTorch | `tch` | Stable | LibTorch bindings. https://docs.rs/crate/tch/0.10.2 |
| PyTorch | `burn` | Active | Rust DL framework. https://docs.rs/burn/latest/index.html |
| PyTorch (backend) | `burn-tch` | Active | Burn backend via LibTorch. https://docs.rs/crate/burn-tch/latest |
| Efficient-KAN | `fekan` | Emerging | Rust KAN implementation. https://docs.rs/fekan |
| NeuralForecast | `ruv-FANN` (Neuro-Divergent) | Emerging | Rust forecasting stack in ruv-FANN monorepo. https://github.com/ruvnet/ruv-FANN |
| TabNet | Manual (burn/tch) | Manual | No verified Rust crate yet. |

### Gradient Boosting
| Python | Rust | Maturity | Notes / Source |
| --- | --- | --- | --- |
| XGBoost | `xgboost` / `xgboostrs` | Early | Rust bindings (early-stage API). https://docs.rs/xgboost/latest/xgboost / https://docs.rs/xgboostrs/latest/xgboostrs/ |
| XGBoost | `xgboost_rs` | Early | Rust bindings. https://docs.rs/xgboost-rs |
| LightGBM | `lightgbm` / `lightgbm2` | Early | Rust bindings. https://docs.rs/lightgbm / https://docs.rs/lightgbm2 |
| LightGBM | `lgbm-sys` | Unofficial | Unofficial LightGBM bindings. https://lib.rs/crates/lgbm-sys |
| CatBoost | `catboost-rs` | Unofficial | Unofficial CatBoost bindings. https://docs.rs/crate/catboost-rs/latest |
| CatBoost (inference) | `catboost` | Limited | Inference-only crate. https://docs.rs/catboost/latest/catboost/ |

### RL / Evolution / HPO
| Python | Rust | Maturity | Notes / Source |
| --- | --- | --- | --- |
| Stable-Baselines3 | `sb3-burn` | WIP | SB3-like RL built on Burn (GitHub). https://github.com/will-maclean/sb3-burn |
| CMA-ES | `cmaes` | Stable | CMA-ES optimizer. https://docs.rs/cmaes/latest/cmaes/ |
| Neuroevolution | `revonet` | Stable | GA + neuroevolution. https://docs.rs/crate/revonet/latest |
| BoTorch/Ax/Optuna | `egobox` + `tpe` | Good | EGObox (EGO/GP toolbox) + TPE optimizer. https://github.com/relf/egobox / https://docs.rs/tpe |

### Infra / Distributed
| Python | Rust | Maturity | Notes / Source |
| --- | --- | --- | --- |
| Ray (single-node) | `tokio` + `rayon` | Stable | Async runtime + data-parallel iterators. https://docs.rs/crate/tokio / https://docs.rs/crate/rayon/latest |
| OpenAI SDK | `async-openai` | Stable | Unofficial OpenAI client. https://docs.rs/crate/async-openai/latest |
| tqdm | `indicatif` | Stable | Progress bars. https://docs.rs/crate/indicatif/latest |
| psutil | `sysinfo` | Stable | System/process info. https://docs.rs/crate/sysinfo/0.20.0 |

## Pending / not yet verified
- GPU data stack: `cudf-cu12`, `cuml-cu12`, `cupy-cuda12x`
- `flash-attn` wheel
- `pytorch-tabnet2`
- `scikit-learn` (possible: linfa/smartcore)
- `scipy`, `numexpr`, `joblib`
- `onnxruntime-gpu` (possible: `ort`)
- `sqlalchemy` (possible: sqlx/sea-orm)
- `pydantic` / `pydantic-settings` (possible: serde/serde_json/config)
- `requests` (possible: reqwest)
- `PyYAML` (possible: serde_yaml)
- `colorama` (possible: colored/termcolor)
- `ray` / `rllib` (beyond single-node)
- `transformers` (possible: candle-transformers)
- `neuralforecast` alternatives beyond ruv-FANN

## Implemented in this repo (Rust so far)
- `rust_core` crate: fast backtest + batch evaluation + TA-Lib feature engine + population evaluator.
- `forex-data` crate: parquet loader, symbol/timeframe discovery, TA-Lib features, multi-timeframe feature join, resampling.
- Rust workspace scaffold (`Cargo.toml`, `crates/forex-cli`, `crates/forex-data`, `crates/forex-models`).
- `forex-search` crate: random + evolutionary search loop powered by `rust_core::evaluate_population_core`.
- `forex-search` crate: strategy GA gene model + discovery portfolio selection + gauntlet + challenge optimizer.
- `forex-search` crate: stop/target estimation + quality scoring + portfolio allocation (Rust native).

## Python modules removed (ported to Rust)
- `src/forex_bot/features/*` (feature pipeline + TA-Lib)
- `src/forex_bot/data/loader.py` (data loading)
- `src/forex_bot/strategy/fast_backtest.py` (backtest)
- `src/forex_bot/strategy/evo_prop.py` (strategy search)
- `src/forex_bot/strategy/rust_bridge.py` (Python bridge)
- `src/forex_bot/strategy/challenge.py` (challenge optimizer)
- `src/forex_bot/strategy/gauntlet.py` (strategy gauntlet)
- `src/forex_bot/strategy/portfolio.py` (portfolio optimizer)
- `src/forex_bot/strategy/quality.py` (quality analyzer + ranker)
- `src/forex_bot/strategy/stop_target.py` (stop/target estimation)
- `src/forex_bot/strategy/genetic.py` (GA gene evolution)
- `src/forex_bot/strategy/discovery.py` (autonomous discovery)
- `src/forex_bot/strategy/discovery_tensor.py` (GPU discovery not ported; CPU discovery now in Rust)

## Next steps
1. Deep learning backend selected: `tch` (LibTorch parity).
2. Phase 1: drop RL models until core + model stack is stable.
3. Fill the Pending list with verified crates or explicit "no equivalent" notes.
