use std::env;
use std::sync::Once;

use ndarray::{Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;
use polars::prelude::*;
use ta_lib_in_rust::indicators::add_technical_indicators;
use std::collections::HashSet;

static RAYON_INIT: Once = Once::new();

fn init_rayon() {
    RAYON_INIT.call_once(|| {
        let threads = env::var("FOREX_BOT_RUST_THREADS")
            .ok()
            .or_else(|| env::var("RAYON_NUM_THREADS").ok())
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0);

        if let Some(n) = threads {
            let _ = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build_global();
        }
    });
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f64;
    let sum: f64 = values.iter().sum();
    let mean = sum / n;
    let mut var = 0.0;
    for v in values {
        let d = v - mean;
        var += d * d;
    }
    let std = (var / n).sqrt();
    (mean, std)
}

pub fn fast_evaluate_strategy_core(
    close_prices: &[f64],
    high_prices: &[f64],
    low_prices: &[f64],
    signals: &[i8],
    month_indices: &[i64],
    day_indices: &[i64],
    sl_pips: f64,
    tp_pips: f64,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> [f64; 11] {
    let n = close_prices.len();
    if n == 0 {
        return [0.0; 11];
    }

    let mut equity = 100000.0_f64;
    let mut peak_equity = 100000.0_f64;

    let mut returns = vec![0.0_f64; n];
    let mut trade_count: usize = 0;
    let mut wins: usize = 0;
    let mut losses: usize = 0;
    let mut gross_profit = 0.0_f64;
    let mut gross_loss = 0.0_f64;

    let mut last_month_val: i64 = -1;
    let mut current_month_pnl = 0.0_f64;
    let mut monthly_pnls = [0.0_f64; 240];
    let mut month_ptr: i64 = -1;

    let mut last_day_val: i64 = -1;
    let mut day_peak = equity;
    let mut day_low = equity;
    let mut max_daily_dd = 0.0_f64;

    let mut in_position: i8 = 0;
    let mut entry_price = 0.0_f64;
    let mut entry_index: i64 = -1;
    let mut trail_price = 0.0_f64;

    let cash_per_pip = pip_value_per_lot;
    let pip_value = if pip_value.abs() < 1e-12 { 1e-12 } else { pip_value };

    for i in 1..n {
        let m_val = if i < month_indices.len() {
            month_indices[i]
        } else {
            last_month_val
        };

        if m_val != last_month_val {
            if last_month_val != -1 {
                month_ptr += 1;
                if month_ptr < 240 {
                    monthly_pnls[month_ptr as usize] = current_month_pnl;
                }
            }
            current_month_pnl = 0.0;
            last_month_val = m_val;
        }

        let d_val = if i < day_indices.len() {
            day_indices[i]
        } else {
            last_day_val
        };

        if d_val != last_day_val {
            if last_day_val != -1 && day_peak > 0.0 {
                let dd = (day_peak - day_low) / day_peak;
                if dd > max_daily_dd {
                    max_daily_dd = dd;
                }
            }
            last_day_val = d_val;
            day_peak = equity;
            day_low = equity;
        }

        if in_position != 0 {
            let current_low = low_prices[i];
            let current_high = high_prices[i];

            let floating_pnl = if in_position == 1 {
                (current_low - entry_price) / pip_value * cash_per_pip
            } else {
                (entry_price - current_high) / pip_value * cash_per_pip
            };

            let current_floating_equity = equity + floating_pnl;
            if current_floating_equity < day_low {
                day_low = current_floating_equity;
            }

            let mut sl_price = 0.0_f64;
            let mut tp_price = 0.0_f64;
            let mut pnl = 0.0_f64;
            let mut exit_signal = false;

            if in_position == 1 {
                sl_price = entry_price - (sl_pips * pip_value);
                tp_price = entry_price + (tp_pips * pip_value);

                if trailing_enabled {
                    let mv = current_high - entry_price;
                    if mv >= (trailing_be_trigger_r * sl_pips * pip_value) {
                        let trail_dist = trailing_atr_multiplier * sl_pips * pip_value;
                        let candidate = current_high - trail_dist;
                        if trail_price == 0.0 || candidate > trail_price {
                            trail_price = candidate;
                        }
                        if trail_price > sl_price {
                            sl_price = trail_price;
                        }
                    }
                }

                if current_low <= sl_price {
                    pnl = (sl_price - entry_price) / pip_value * cash_per_pip;
                    exit_signal = true;
                } else if current_high >= tp_price {
                    pnl = (tp_price - entry_price) / pip_value * cash_per_pip;
                    exit_signal = true;
                }
            } else {
                sl_price = entry_price + (sl_pips * pip_value);
                tp_price = entry_price - (tp_pips * pip_value);

                if trailing_enabled {
                    let mv = entry_price - current_low;
                    if mv >= (trailing_be_trigger_r * sl_pips * pip_value) {
                        let trail_dist = trailing_atr_multiplier * sl_pips * pip_value;
                        let candidate = current_low + trail_dist;
                        if trail_price == 0.0 || candidate < trail_price {
                            trail_price = candidate;
                        }
                        if trail_price < sl_price {
                            sl_price = trail_price;
                        }
                    }
                }

                if current_high >= sl_price {
                    pnl = (entry_price - sl_price) / pip_value * cash_per_pip;
                    exit_signal = true;
                } else if current_low <= tp_price {
                    pnl = (entry_price - tp_price) / pip_value * cash_per_pip;
                    exit_signal = true;
                }
            }

            if !exit_signal && max_hold_bars > 0 && entry_index >= 0 {
                if (i as i64 - entry_index) as usize >= max_hold_bars {
                    if in_position == 1 {
                        pnl = (close_prices[i] - entry_price) / pip_value * cash_per_pip;
                    } else {
                        pnl = (entry_price - close_prices[i]) / pip_value * cash_per_pip;
                    }
                    exit_signal = true;
                }
            }

            if !exit_signal {
                let sig = signals[i];
                if in_position == 1 && sig == -1 {
                    pnl = (close_prices[i] - entry_price) / pip_value * cash_per_pip;
                    exit_signal = true;
                } else if in_position == -1 && sig == 1 {
                    pnl = (entry_price - close_prices[i]) / pip_value * cash_per_pip;
                    exit_signal = true;
                }
            }

            if exit_signal {
                let lot_size = 1.0_f64;
                let cost_in_pips = spread_pips;
                let commission_in_cash = commission_per_trade * lot_size * 2.0;
                let total_cost = (cost_in_pips * cash_per_pip) + commission_in_cash;
                pnl -= total_cost;

                equity += pnl;
                current_month_pnl += pnl;
                if trade_count < returns.len() {
                    returns[trade_count] = pnl;
                }
                trade_count += 1;
                in_position = 0;
                entry_index = -1;
                trail_price = 0.0;

                if pnl > 0.0 {
                    wins += 1;
                    gross_profit += pnl;
                } else {
                    losses += 1;
                    gross_loss += pnl.abs();
                }

                if equity > peak_equity {
                    peak_equity = equity;
                }
                if equity > day_peak {
                    day_peak = equity;
                }
                if equity < day_low {
                    day_low = equity;
                }
            }
        }

        if in_position == 0 {
            let sig = signals[i];
            if sig == 1 {
                in_position = 1;
                entry_price = close_prices[i] + (spread_pips * pip_value);
                entry_index = i as i64;
                trail_price = 0.0;
            } else if sig == -1 {
                in_position = -1;
                entry_price = close_prices[i];
                entry_index = i as i64;
                trail_price = 0.0;
            }
        }
    }

    if month_ptr < 239 {
        monthly_pnls[(month_ptr + 1) as usize] = current_month_pnl;
    }

    if last_day_val != -1 && day_peak > 0.0 {
        let dd = (day_peak - day_low) / day_peak;
        if dd > max_daily_dd {
            max_daily_dd = dd;
        }
    }

    if trade_count == 0 {
        return [0.0; 11];
    }

    let trade_rets = &returns[..trade_count];
    let (avg_trade, std_trade) = mean_std(trade_rets);
    let sharpe = if std_trade > 0.0 {
        (avg_trade / std_trade) * (trade_count as f64).sqrt()
    } else {
        0.0
    };

    let downside: Vec<f64> = trade_rets.iter().cloned().filter(|v| *v < 0.0).collect();
    let std_down = if downside.is_empty() {
        1.0
    } else {
        mean_std(&downside).1
    };
    let sortino = if std_down > 0.0 {
        (avg_trade / std_down) * (trade_count as f64).sqrt()
    } else {
        0.0
    };

    let mut equity_curve = vec![0.0_f64; trade_count + 1];
    equity_curve[0] = 100000.0;
    let mut curr = 100000.0_f64;
    let mut max_dd = 0.0_f64;
    let mut peak = 100000.0_f64;
    for (k, ret) in trade_rets.iter().enumerate() {
        curr += ret;
        equity_curve[k + 1] = curr;
        if curr > peak {
            peak = curr;
        }
        let dd = (peak - curr) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }

    let mut r_squared = 0.0_f64;
    if trade_count > 2 {
        let n_points = (trade_count + 1) as f64;
        let mut sum_x = 0.0_f64;
        let mut sum_y = 0.0_f64;
        let mut sum_xy = 0.0_f64;
        let mut sum_xx = 0.0_f64;
        let mut sum_yy = 0.0_f64;
        for (i, y) in equity_curve.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += *y;
            sum_xy += x * y;
            sum_xx += x * x;
            sum_yy += y * y;
        }
        let numerator = (n_points * sum_xy) - (sum_x * sum_y);
        let denominator_x = (n_points * sum_xx) - (sum_x * sum_x);
        let denominator_y = (n_points * sum_yy) - (sum_y * sum_y);
        if denominator_x > 0.0 && denominator_y > 0.0 {
            let r = numerator / (denominator_x * denominator_y).sqrt();
            r_squared = r * r;
        }
    }

    let active_months_end = (month_ptr + 2).clamp(0, 240) as usize;
    let active_months = &monthly_pnls[..active_months_end];
    let negative_months = active_months.iter().filter(|v| **v < 0.0).count();

    let mut consistency_score = r_squared;
    if negative_months > 0 {
        consistency_score *= 1.0 - (negative_months as f64 * 0.1);
        if consistency_score < 0.0 {
            consistency_score = 0.0;
        }
    } else if active_months.len() > 1 {
        consistency_score += 0.2;
    }

    let net_profit = equity - 100000.0;
    let win_rate = wins as f64 / trade_count as f64;
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        99.0
    };
    let expectancy = avg_trade;
    let sqn = if std_trade > 0.0 {
        (avg_trade / std_trade) * (trade_count as f64).sqrt()
    } else {
        0.0
    };

    [
        net_profit,
        sharpe,
        sortino,
        max_dd,
        win_rate,
        profit_factor,
        expectancy,
        sqn,
        trade_count as f64,
        consistency_score,
        max_daily_dd,
    ]
}

fn evaluate_population_impl(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    month_indices: &[i64],
    day_indices: &[i64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    ob_arr: &[i8],
    fvg_arr: &[i8],
    liq_arr: &[i8],
    trend_arr: &[i8],
    premium_arr: &[i8],
    inducement_arr: &[i8],
    use_ob_arr: &[i8],
    use_fvg_arr: &[i8],
    use_liq_arr: &[i8],
    use_mtf_arr: &[i8],
    use_premium_arr: &[i8],
    use_inducement_arr: &[i8],
    smc_gate_threshold: f32,
    smc_weight_ob: f32,
    smc_weight_fvg: f32,
    smc_weight_liq: f32,
    smc_weight_mtf: f32,
    smc_weight_premium: f32,
    smc_weight_inducement: f32,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> Result<Vec<[f64; 11]>, String> {
    let n_genes = long_thr.len();
    if gene_offsets.len() != n_genes + 1 {
        return Err("gene_offsets length must be n_genes+1".to_string());
    }
    if short_thr.len() != n_genes || sl_pips.len() != n_genes || tp_pips.len() != n_genes {
        return Err("per-gene arrays must match gene count".to_string());
    }
    if close.len() != high.len() || close.len() != low.len() {
        return Err("OHLC arrays must match length".to_string());
    }
    let n_samples = close.len();
    if indicators.shape().len() != 2 || indicators.shape()[1] != n_samples {
        return Err("indicator_matrix shape mismatch".to_string());
    }
    if use_ob_arr.len() != n_genes
        || use_fvg_arr.len() != n_genes
        || use_liq_arr.len() != n_genes
        || use_mtf_arr.len() != n_genes
        || use_premium_arr.len() != n_genes
        || use_inducement_arr.len() != n_genes
    {
        return Err("per-gene gate flags must match gene count".to_string());
    }
    if ob_arr.len() != n_samples
        || fvg_arr.len() != n_samples
        || liq_arr.len() != n_samples
        || trend_arr.len() != n_samples
        || premium_arr.len() != n_samples
        || inducement_arr.len() != n_samples
    {
        return Err("SMC arrays must match OHLC length".to_string());
    }

    let results: Vec<[f64; 11]> = (0..n_genes)
        .into_par_iter()
        .map(|g| {
            let start = gene_offsets[g] as usize;
            let end = gene_offsets[g + 1] as usize;
            let mut combined = vec![0.0_f32; n_samples];
            if end > start {
                for idx in start..end {
                    let ind_idx = gene_indices[idx] as usize;
                    let weight = gene_weights[idx];
                    if ind_idx >= indicators.shape()[0] {
                        continue;
                    }
                    let row = indicators.row(ind_idx);
                    for (i, v) in row.iter().enumerate() {
                        combined[i] += weight * (*v);
                    }
                }
            }

            let lt = long_thr[g];
            let st = short_thr[g];
            let mut signals = vec![0i8; n_samples];
            for i in 0..n_samples {
                let v = combined[i];
                if v >= lt {
                    signals[i] = 1;
                } else if v <= st {
                    signals[i] = -1;
                }
            }

            let use_ob_g = use_ob_arr[g] != 0;
            let use_fvg_g = use_fvg_arr[g] != 0;
            let use_liq_g = use_liq_arr[g] != 0;
            let use_mtf_g = use_mtf_arr[g] != 0;
            let use_premium_g = use_premium_arr[g] != 0;
            let use_inducement_g = use_inducement_arr[g] != 0;

            let mut active_weight_sum = 0.0_f32;
            if use_ob_g {
                active_weight_sum += smc_weight_ob;
            }
            if use_fvg_g {
                active_weight_sum += smc_weight_fvg;
            }
            if use_liq_g {
                active_weight_sum += smc_weight_liq;
            }
            if use_mtf_g {
                active_weight_sum += smc_weight_mtf;
            }
            if use_premium_g {
                active_weight_sum += smc_weight_premium;
            }
            if use_inducement_g {
                active_weight_sum += smc_weight_inducement;
            }

            if active_weight_sum > 0.0 {
                let gate_threshold = smc_gate_threshold.min(active_weight_sum);
                for i in 0..n_samples {
                    let dir = signals[i];
                    if dir == 0 {
                        continue;
                    }
                    let mut score = 0.0_f32;
                    if use_ob_g && ob_arr[i] == dir {
                        score += smc_weight_ob;
                    }
                    if use_fvg_g && fvg_arr[i] == dir {
                        score += smc_weight_fvg;
                    }
                    if use_liq_g && liq_arr[i] == dir {
                        score += smc_weight_liq;
                    }
                    if use_mtf_g && trend_arr[i] == dir {
                        score += smc_weight_mtf;
                    }
                    if use_premium_g && premium_arr[i] == dir {
                        score += smc_weight_premium;
                    }
                    if use_inducement_g && inducement_arr[i] == 1 {
                        score += smc_weight_inducement;
                    }
                    if score < gate_threshold {
                        signals[i] = 0;
                    }
                }
            }

            fast_evaluate_strategy_core(
                close,
                high,
                low,
                &signals,
                month_indices,
                day_indices,
                sl_pips[g],
                tp_pips[g],
                max_hold_bars,
                trailing_enabled,
                trailing_atr_multiplier,
                trailing_be_trigger_r,
                pip_value,
                spread_pips,
                commission_per_trade,
                pip_value_per_lot,
            )
        })
        .collect();

    Ok(results)
}

pub fn evaluate_population_core(
    close: &[f64],
    high: &[f64],
    low: &[f64],
    indicators: ArrayView2<'_, f32>,
    gene_offsets: &[i32],
    gene_indices: &[i32],
    gene_weights: &[f32],
    long_thr: &[f32],
    short_thr: &[f32],
    month_indices: &[i64],
    day_indices: &[i64],
    sl_pips: &[f64],
    tp_pips: &[f64],
    ob_arr: &[i8],
    fvg_arr: &[i8],
    liq_arr: &[i8],
    trend_arr: &[i8],
    premium_arr: &[i8],
    inducement_arr: &[i8],
    use_ob_arr: &[i8],
    use_fvg_arr: &[i8],
    use_liq_arr: &[i8],
    use_mtf_arr: &[i8],
    use_premium_arr: &[i8],
    use_inducement_arr: &[i8],
    smc_gate_threshold: f32,
    smc_weight_ob: f32,
    smc_weight_fvg: f32,
    smc_weight_liq: f32,
    smc_weight_mtf: f32,
    smc_weight_premium: f32,
    smc_weight_inducement: f32,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> Result<Vec<[f64; 11]>, String> {
    init_rayon();
    evaluate_population_impl(
        close,
        high,
        low,
        indicators,
        gene_offsets,
        gene_indices,
        gene_weights,
        long_thr,
        short_thr,
        month_indices,
        day_indices,
        sl_pips,
        tp_pips,
        ob_arr,
        fvg_arr,
        liq_arr,
        trend_arr,
        premium_arr,
        inducement_arr,
        use_ob_arr,
        use_fvg_arr,
        use_liq_arr,
        use_mtf_arr,
        use_premium_arr,
        use_inducement_arr,
        smc_gate_threshold,
        smc_weight_ob,
        smc_weight_fvg,
        smc_weight_liq,
        smc_weight_mtf,
        smc_weight_premium,
        smc_weight_inducement,
        max_hold_bars,
        trailing_enabled,
        trailing_atr_multiplier,
        trailing_be_trigger_r,
        pip_value,
        spread_pips,
        commission_per_trade,
        pip_value_per_lot,
    )
}

#[pyfunction]
#[pyo3(signature = (
    close_prices,
    high_prices,
    low_prices,
    signals,
    month_indices,
    day_indices,
    sl_pips,
    tp_pips,
    ob,
    fvg,
    liq,
    trend,
    premium_discount,
    inducement,
    use_ob,
    use_fvg,
    use_liq,
    use_mtf,
    use_premium,
    use_inducement,
    smc_gate_threshold,
    smc_weight_ob,
    smc_weight_fvg,
    smc_weight_liq,
    smc_weight_mtf,
    smc_weight_premium,
    smc_weight_inducement,
    max_hold_bars=0,
    trailing_enabled=false,
    trailing_atr_multiplier=1.0,
    trailing_be_trigger_r=1.0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission_per_trade=0.0,
    pip_value_per_lot=10.0
))]
fn fast_evaluate_strategy(
    py: Python<'_>,
    close_prices: PyReadonlyArray1<f64>,
    high_prices: PyReadonlyArray1<f64>,
    low_prices: PyReadonlyArray1<f64>,
    signals: PyReadonlyArray1<i8>,
    month_indices: PyReadonlyArray1<i64>,
    day_indices: PyReadonlyArray1<i64>,
    sl_pips: f64,
    tp_pips: f64,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> PyResult<Py<PyArray1<f64>>> {
    init_rayon();

    let close = close_prices.as_slice().map_err(|_| PyValueError::new_err("close_prices must be contiguous"))?;
    let high = high_prices.as_slice().map_err(|_| PyValueError::new_err("high_prices must be contiguous"))?;
    let low = low_prices.as_slice().map_err(|_| PyValueError::new_err("low_prices must be contiguous"))?;
    let sig = signals.as_slice().map_err(|_| PyValueError::new_err("signals must be contiguous"))?;
    let months = month_indices.as_slice().map_err(|_| PyValueError::new_err("month_indices must be contiguous"))?;
    let days = day_indices.as_slice().map_err(|_| PyValueError::new_err("day_indices must be contiguous"))?;

    if close.len() != high.len() || close.len() != low.len() || close.len() != sig.len() {
        return Err(PyValueError::new_err("Input arrays must have matching lengths"));
    }

    let out = fast_evaluate_strategy_core(
        close,
        high,
        low,
        sig,
        months,
        days,
        sl_pips,
        tp_pips,
        max_hold_bars,
        trailing_enabled,
        trailing_atr_multiplier,
        trailing_be_trigger_r,
        pip_value,
        spread_pips,
        commission_per_trade,
        pip_value_per_lot,
    );

    Ok(PyArray1::from_vec(py, out.to_vec()).to_owned())
}

#[pyfunction]
#[pyo3(signature = (
    close_prices_batch,
    high_prices_batch,
    low_prices_batch,
    signals_batch,
    month_indices_batch,
    day_indices_batch,
    sl_pips,
    tp_pips,
    max_hold_bars=0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission=0.0,
    pip_val_lot=10.0
))]
fn batch_evaluate_strategies(
    py: Python<'_>,
    close_prices_batch: PyReadonlyArray2<f64>,
    high_prices_batch: PyReadonlyArray2<f64>,
    low_prices_batch: PyReadonlyArray2<f64>,
    signals_batch: PyReadonlyArray2<i8>,
    month_indices_batch: PyReadonlyArray2<i64>,
    day_indices_batch: PyReadonlyArray2<i64>,
    sl_pips: f64,
    tp_pips: f64,
    max_hold_bars: usize,
    pip_value: f64,
    spread_pips: f64,
    commission: f64,
    pip_val_lot: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    init_rayon();

    let close = close_prices_batch.as_array();
    let high = high_prices_batch.as_array();
    let low = low_prices_batch.as_array();
    let sig = signals_batch.as_array();
    let months = month_indices_batch.as_array();
    let days = day_indices_batch.as_array();

    let n_strats = close.shape()[0];
    if n_strats == 0 {
        let out = Array2::<f64>::zeros((0, 11));
        return Ok(PyArray2::from_owned_array(py, out).to_owned());
    }

    if high.shape() != close.shape()
        || low.shape() != close.shape()
        || sig.shape() != close.shape()
        || months.shape() != close.shape()
        || days.shape() != close.shape()
    {
        return Err(PyValueError::new_err("Batch arrays must have matching shapes"));
    }

    let results: Vec<[f64; 11]> = (0..n_strats)
        .into_par_iter()
        .map(|s| {
            let close_row = close.row(s).as_slice().expect("close row contiguous");
            let high_row = high.row(s).as_slice().expect("high row contiguous");
            let low_row = low.row(s).as_slice().expect("low row contiguous");
            let sig_row = sig.row(s).as_slice().expect("signals row contiguous");
            let month_row = months.row(s).as_slice().expect("month row contiguous");
            let day_row = days.row(s).as_slice().expect("day row contiguous");

            fast_evaluate_strategy_core(
                close_row,
                high_row,
                low_row,
                sig_row,
                month_row,
                day_row,
                sl_pips,
                tp_pips,
                max_hold_bars,
                false,
                1.0,
                1.0,
                pip_value,
                spread_pips,
                commission,
                pip_val_lot,
            )
        })
        .collect();

    let mut out = Array2::<f64>::zeros((n_strats, 11));
    for (i, metrics) in results.iter().enumerate() {
        for j in 0..11 {
            out[(i, j)] = metrics[j];
        }
    }

    Ok(PyArray2::from_owned_array(py, out).to_owned())
}

#[pyfunction]
#[pyo3(signature = (
    close_prices,
    high_prices,
    low_prices,
    indicator_matrix,
    gene_offsets,
    gene_indices,
    gene_weights,
    long_thresholds,
    short_thresholds,
    month_indices,
    day_indices,
    sl_pips,
    tp_pips,
    max_hold_bars=0,
    trailing_enabled=false,
    trailing_atr_multiplier=1.0,
    trailing_be_trigger_r=1.0,
    pip_value=0.0001,
    spread_pips=1.5,
    commission_per_trade=0.0,
    pip_value_per_lot=10.0
))]
fn evaluate_population(
    py: Python<'_>,
    close_prices: PyReadonlyArray1<f64>,
    high_prices: PyReadonlyArray1<f64>,
    low_prices: PyReadonlyArray1<f64>,
    indicator_matrix: PyReadonlyArray2<f32>,
    gene_offsets: PyReadonlyArray1<i32>,
    gene_indices: PyReadonlyArray1<i32>,
    gene_weights: PyReadonlyArray1<f32>,
    long_thresholds: PyReadonlyArray1<f32>,
    short_thresholds: PyReadonlyArray1<f32>,
    month_indices: PyReadonlyArray1<i64>,
    day_indices: PyReadonlyArray1<i64>,
    sl_pips: PyReadonlyArray1<f64>,
    tp_pips: PyReadonlyArray1<f64>,
    ob: PyReadonlyArray1<i8>,
    fvg: PyReadonlyArray1<i8>,
    liq: PyReadonlyArray1<i8>,
    trend: PyReadonlyArray1<i8>,
    premium_discount: PyReadonlyArray1<i8>,
    inducement: PyReadonlyArray1<i8>,
    use_ob: PyReadonlyArray1<i8>,
    use_fvg: PyReadonlyArray1<i8>,
    use_liq: PyReadonlyArray1<i8>,
    use_mtf: PyReadonlyArray1<i8>,
    use_premium: PyReadonlyArray1<i8>,
    use_inducement: PyReadonlyArray1<i8>,
    smc_gate_threshold: f32,
    smc_weight_ob: f32,
    smc_weight_fvg: f32,
    smc_weight_liq: f32,
    smc_weight_mtf: f32,
    smc_weight_premium: f32,
    smc_weight_inducement: f32,
    max_hold_bars: usize,
    trailing_enabled: bool,
    trailing_atr_multiplier: f64,
    trailing_be_trigger_r: f64,
    pip_value: f64,
    spread_pips: f64,
    commission_per_trade: f64,
    pip_value_per_lot: f64,
) -> PyResult<Py<PyArray2<f64>>> {
    init_rayon();

    let close = close_prices
        .as_slice()
        .map_err(|_| PyValueError::new_err("close_prices must be contiguous"))?;
    let high = high_prices
        .as_slice()
        .map_err(|_| PyValueError::new_err("high_prices must be contiguous"))?;
    let low = low_prices
        .as_slice()
        .map_err(|_| PyValueError::new_err("low_prices must be contiguous"))?;
    let indicators = indicator_matrix.as_array();
    let offsets = gene_offsets
        .as_slice()
        .map_err(|_| PyValueError::new_err("gene_offsets must be contiguous"))?;
    let indices = gene_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("gene_indices must be contiguous"))?;
    let weights = gene_weights
        .as_slice()
        .map_err(|_| PyValueError::new_err("gene_weights must be contiguous"))?;
    let long_thr = long_thresholds
        .as_slice()
        .map_err(|_| PyValueError::new_err("long_thresholds must be contiguous"))?;
    let short_thr = short_thresholds
        .as_slice()
        .map_err(|_| PyValueError::new_err("short_thresholds must be contiguous"))?;
    let months = month_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("month_indices must be contiguous"))?;
    let days = day_indices
        .as_slice()
        .map_err(|_| PyValueError::new_err("day_indices must be contiguous"))?;
    let sl_arr = sl_pips
        .as_slice()
        .map_err(|_| PyValueError::new_err("sl_pips must be contiguous"))?;
    let tp_arr = tp_pips
        .as_slice()
        .map_err(|_| PyValueError::new_err("tp_pips must be contiguous"))?;
    let ob_arr = ob
        .as_slice()
        .map_err(|_| PyValueError::new_err("ob must be contiguous"))?;
    let fvg_arr = fvg
        .as_slice()
        .map_err(|_| PyValueError::new_err("fvg must be contiguous"))?;
    let liq_arr = liq
        .as_slice()
        .map_err(|_| PyValueError::new_err("liq must be contiguous"))?;
    let trend_arr = trend
        .as_slice()
        .map_err(|_| PyValueError::new_err("trend must be contiguous"))?;
    let premium_arr = premium_discount
        .as_slice()
        .map_err(|_| PyValueError::new_err("premium_discount must be contiguous"))?;
    let inducement_arr = inducement
        .as_slice()
        .map_err(|_| PyValueError::new_err("inducement must be contiguous"))?;
    let use_ob_arr = use_ob
        .as_slice()
        .map_err(|_| PyValueError::new_err("use_ob must be contiguous"))?;
    let use_fvg_arr = use_fvg
        .as_slice()
        .map_err(|_| PyValueError::new_err("use_fvg must be contiguous"))?;
    let use_liq_arr = use_liq
        .as_slice()
        .map_err(|_| PyValueError::new_err("use_liq must be contiguous"))?;
    let use_mtf_arr = use_mtf
        .as_slice()
        .map_err(|_| PyValueError::new_err("use_mtf must be contiguous"))?;
    let use_premium_arr = use_premium
        .as_slice()
        .map_err(|_| PyValueError::new_err("use_premium must be contiguous"))?;
    let use_inducement_arr = use_inducement
        .as_slice()
        .map_err(|_| PyValueError::new_err("use_inducement must be contiguous"))?;

    let n_genes = long_thr.len();
    if offsets.len() != n_genes + 1 {
        return Err(PyValueError::new_err("gene_offsets length must be n_genes+1"));
    }
    if short_thr.len() != n_genes || sl_arr.len() != n_genes || tp_arr.len() != n_genes {
        return Err(PyValueError::new_err("per-gene arrays must match gene count"));
    }
    if use_ob_arr.len() != n_genes
        || use_fvg_arr.len() != n_genes
        || use_liq_arr.len() != n_genes
        || use_mtf_arr.len() != n_genes
        || use_premium_arr.len() != n_genes
        || use_inducement_arr.len() != n_genes
    {
        return Err(PyValueError::new_err(
            "per-gene gate flags must match gene count",
        ));
    }
    if close.len() != high.len() || close.len() != low.len() {
        return Err(PyValueError::new_err("OHLC arrays must match length"));
    }
    let n_samples = close.len();
    if indicators.shape().len() != 2 || indicators.shape()[1] != n_samples {
        return Err(PyValueError::new_err("indicator_matrix shape mismatch"));
    }
    if ob_arr.len() != n_samples
        || fvg_arr.len() != n_samples
        || liq_arr.len() != n_samples
        || trend_arr.len() != n_samples
        || premium_arr.len() != n_samples
        || inducement_arr.len() != n_samples
    {
        return Err(PyValueError::new_err("SMC arrays must match OHLC length"));
    }

    let results = evaluate_population_core(
        close,
        high,
        low,
        indicators,
        offsets,
        indices,
        weights,
        long_thr,
        short_thr,
        months,
        days,
        sl_arr,
        tp_arr,
        ob_arr,
        fvg_arr,
        liq_arr,
        trend_arr,
        premium_arr,
        inducement_arr,
        use_ob_arr,
        use_fvg_arr,
        use_liq_arr,
        use_mtf_arr,
        use_premium_arr,
        use_inducement_arr,
        smc_gate_threshold,
        smc_weight_ob,
        smc_weight_fvg,
        smc_weight_liq,
        smc_weight_mtf,
        smc_weight_premium,
        smc_weight_inducement,
        max_hold_bars,
        trailing_enabled,
        trailing_atr_multiplier,
        trailing_be_trigger_r,
        pip_value,
        spread_pips,
        commission_per_trade,
        pip_value_per_lot,
    )
    .map_err(|err| PyValueError::new_err(err))?;

    let mut out = Array2::<f64>::zeros((n_genes, 11));
    for (i, metrics) in results.iter().enumerate() {
        for j in 0..11 {
            out[(i, j)] = metrics[j];
        }
    }

    Ok(PyArray2::from_owned_array(py, out).to_owned())
}

fn series_to_vec_f32(series: &Series, n_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_rows];
    let casted = series.cast(&DataType::Float64).unwrap_or_else(|_| series.clone());
    if let Ok(chunked) = casted.f64() {
        for (i, v) in chunked.into_iter().enumerate().take(n_rows) {
            out[i] = v.unwrap_or(0.0) as f32;
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (open_prices, high_prices, low_prices, close_prices, volume=None))]
fn compute_talib_features(
    py: Python<'_>,
    open_prices: PyReadonlyArray1<f64>,
    high_prices: PyReadonlyArray1<f64>,
    low_prices: PyReadonlyArray1<f64>,
    close_prices: PyReadonlyArray1<f64>,
    volume: Option<PyReadonlyArray1<f64>>,
) -> PyResult<(Py<PyArray2<f32>>, Vec<String>)> {
    init_rayon();

    let open = open_prices
        .as_slice()
        .map_err(|_| PyValueError::new_err("open must be contiguous"))?;
    let high = high_prices
        .as_slice()
        .map_err(|_| PyValueError::new_err("high must be contiguous"))?;
    let low = low_prices
        .as_slice()
        .map_err(|_| PyValueError::new_err("low must be contiguous"))?;
    let close = close_prices
        .as_slice()
        .map_err(|_| PyValueError::new_err("close must be contiguous"))?;

    if open.len() != high.len() || open.len() != low.len() || open.len() != close.len() {
        return Err(PyValueError::new_err("OHLC arrays must have matching lengths"));
    }

    let n_rows = close.len();
    let volume_vec: Vec<f64> = if let Some(vol) = volume.as_ref() {
        let vol_slice = vol
            .as_slice()
            .map_err(|_| PyValueError::new_err("volume must be contiguous"))?;
        if vol_slice.len() != n_rows {
            return Err(PyValueError::new_err("volume length must match close"));
        }
        vol_slice.to_vec()
    } else {
        vec![0.0_f64; n_rows]
    };

    let mut df = DataFrame::new(vec![
        Series::new("open", open.to_vec()),
        Series::new("high", high.to_vec()),
        Series::new("low", low.to_vec()),
        Series::new("close", close.to_vec()),
        Series::new("volume", volume_vec),
    ])
    .map_err(|e| PyValueError::new_err(format!("polars DataFrame build failed: {e}")))?;

    let original_cols: HashSet<String> = df
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();

    let df = add_technical_indicators(&mut df)
        .map_err(|e| PyValueError::new_err(format!("ta-lib-in-rust failed: {e}")))?;

    let mut names: Vec<String> = Vec::new();
    let mut columns: Vec<Vec<f32>> = Vec::new();

    for series in df.get_columns() {
        let name = series.name();
        if original_cols.contains(name) {
            continue;
        }
        let mut col_name = String::from("ta_");
        col_name.push_str(name);
        names.push(col_name);
        columns.push(series_to_vec_f32(series, n_rows));
    }

    let n_cols = columns.len();
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    for (col_idx, vals) in columns.iter().enumerate() {
        let len = vals.len().min(n_rows);
        for i in 0..len {
            out[(i, col_idx)] = vals[i];
        }
    }

    Ok((PyArray2::from_owned_array(py, out).to_owned(), names))
}

#[pymodule]
fn forex_core(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fast_evaluate_strategy, m)?)?;
    m.add_function(wrap_pyfunction!(batch_evaluate_strategies, m)?)?;
    m.add_function(wrap_pyfunction!(compute_talib_features, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_population, m)?)?;
    Ok(())
}
