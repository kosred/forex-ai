use anyhow::Result;
use ndarray::{Array1, Array2};
use polars::prelude::*;
use std::collections::HashMap;

/// Convert class probabilities to signals [-1, 0, 1].
/// Assumes [Neutral, Buy, Sell] convention at indices [0, 1, 2].
pub fn probs_to_signals(probs: &Array2<f32>) -> Array1<i32> {
    let mut signals = Array1::zeros(probs.nrows());
    
    for i in 0..probs.nrows() {
        let row = probs.row(i);
        let mut max_val = row[0];
        let mut max_idx = 0;
        
        for (j, &val) in row.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = j;
            }
        }
        
        signals[i] = match max_idx {
            1 => 1,   // Buy
            2 => -1,  // Sell
            _ => 0,   // Neutral
        };
    }
    
    signals
}

/// Simple backtest to evaluate model performance on historical data.
pub fn simple_backtest(df: &DataFrame, signals: &Array1<i32>) -> Result<HashMap<String, f64>> {
    if df.height() == 0 || signals.len() != df.height() {
        return Ok(HashMap::new());
    }

    let close_col = df.column("close")?.as_materialized_series();
    let close: Vec<f64> = close_col.cast(&DataType::Float64)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(0.0))
        .collect();

    let mut pnl = Vec::new();
    let mut n_trades = 0;

    for i in 0..close.len() - 1 {
        let sig = signals[i];
        let curr_close = close[i];
        let next_close = close[i + 1];
        let ret = next_close - curr_close;

        if sig == 0 {
            pnl.push(0.0);
        } else {
            n_trades += 1;
            if sig == 1 {
                pnl.push(if ret > 0.0 { 1.0 } else { -1.0 });
            } else if sig == -1 {
                pnl.push(if ret < 0.0 { 1.0 } else { -1.0 });
            }
        }
    }

    let pnl_score: f64 = pnl.iter().sum();
    let win_rate = if !pnl.is_empty() {
        pnl.iter().filter(|&&v| v > 0.0).count() as f64 / pnl.len() as f64
    } else {
        0.0
    };

    let mut results = HashMap::new();
    results.insert("pnl_score".to_string(), pnl_score);
    results.insert("win_rate".to_string(), win_rate);
    results.insert("trades".to_string(), n_trades as f64);

    Ok(results)
}
