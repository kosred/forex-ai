use std::f64::consts::LN_2;

#[derive(Debug, Clone, Copy)]
pub struct VolEnsembleWeights {
    pub yang_zhang: f64,
    pub garman_klass: f64,
    pub rogers_satchell: f64,
    pub parkinson: f64,
}

impl VolEnsembleWeights {
    fn normalize(&self) -> Option<[f64; 4]> {
        let mut vals = [
            self.yang_zhang.max(0.0),
            self.garman_klass.max(0.0),
            self.rogers_satchell.max(0.0),
            self.parkinson.max(0.0),
        ];
        let total: f64 = vals.iter().sum();
        if total <= 0.0 {
            return None;
        }
        for v in &mut vals {
            *v /= total;
        }
        Some(vals)
    }
}

#[derive(Debug, Clone)]
pub struct StopTargetSettings {
    pub vol_estimator: String,
    pub vol_window: usize,
    pub ewma_lambda: f64,
    pub vol_horizon_bars: usize,
    pub tail_window: usize,
    pub tail_alpha: f64,
    pub tail_step: usize,
    pub tail_max_bars: usize,
    pub stop_k_vol: f64,
    pub stop_k_tail: f64,
    pub meta_label_min_dist: f64,
    pub regime_adx_trend: f64,
    pub regime_adx_range: f64,
    pub hurst_window: usize,
    pub hurst_trend: f64,
    pub hurst_range: f64,
    pub rr_trend: f64,
    pub rr_range: f64,
    pub rr_neutral: f64,
    pub ema_fast_period: usize,
    pub ema_slow_period: usize,
    pub atr_period: usize,
    pub weights: Option<VolEnsembleWeights>,
    pub weights_trend: Option<VolEnsembleWeights>,
    pub weights_range: Option<VolEnsembleWeights>,
}

impl Default for StopTargetSettings {
    fn default() -> Self {
        let rr_trend = 2.5;
        let rr_range = 1.5;
        Self {
            vol_estimator: "yang_zhang".to_string(),
            vol_window: 50,
            ewma_lambda: 0.94,
            vol_horizon_bars: 5,
            tail_window: 100,
            tail_alpha: 0.975,
            tail_step: 5,
            tail_max_bars: 300_000,
            stop_k_vol: 1.0,
            stop_k_tail: 1.25,
            meta_label_min_dist: 0.0,
            regime_adx_trend: 25.0,
            regime_adx_range: 20.0,
            hurst_window: 100,
            hurst_trend: 0.55,
            hurst_range: 0.45,
            rr_trend,
            rr_range,
            rr_neutral: (rr_trend + rr_range) / 2.0,
            ema_fast_period: 20,
            ema_slow_period: 50,
            atr_period: 14,
            weights: None,
            weights_trend: None,
            weights_range: None,
        }
    }
}

fn safe_log(v: f64) -> f64 {
    v.max(1e-12).ln()
}

fn rolling_mean(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if window <= 1 {
        return values.to_vec();
    }
    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0;
    for i in 0..n {
        sum += values[i];
        if i + 1 >= window {
            out[i] = sum / window as f64;
            sum -= values[i + 1 - window];
        }
    }
    out
}

fn rolling_var(values: &[f64], window: usize) -> Vec<f64> {
    let n = values.len();
    if window <= 1 {
        return vec![0.0; n];
    }
    let mut out = vec![f64::NAN; n];
    let mut sum = 0.0;
    let mut sumsq = 0.0;
    for i in 0..n {
        sum += values[i];
        sumsq += values[i] * values[i];
        if i + 1 >= window {
            let mean = sum / window as f64;
            let var = (sumsq - sum * mean) / (window as f64 - 1.0);
            out[i] = var.max(0.0);
            sum -= values[i + 1 - window];
            sumsq -= values[i + 1 - window] * values[i + 1 - window];
        }
    }
    out
}

fn vol_parkinson(high: &[f64], low: &[f64]) -> Vec<f64> {
    high
        .iter()
        .zip(low.iter())
        .map(|(h, l)| {
            let hl = safe_log(*h) - safe_log(*l);
            (hl * hl) / (4.0 * LN_2)
        })
        .collect()
}

fn vol_rogers_satchell(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        let ho = safe_log(high[i]) - safe_log(open[i]);
        let hc = safe_log(high[i]) - safe_log(close[i]);
        let lo = safe_log(low[i]) - safe_log(open[i]);
        let lc = safe_log(low[i]) - safe_log(close[i]);
        out.push((ho * hc) + (lo * lc));
    }
    out
}

fn vol_garman_klass(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(close.len());
    let c1 = 2.0 * LN_2 - 1.0;
    for i in 0..close.len() {
        let h = safe_log(high[i]);
        let l = safe_log(low[i]);
        let o = safe_log(open[i]);
        let c = safe_log(close[i]);
        let hl = h - l;
        let co = c - o;
        out.push(0.5 * (hl * hl) - c1 * (co * co));
    }
    out
}

fn vol_yang_zhang(open: &[f64], high: &[f64], low: &[f64], close: &[f64], window: usize) -> Vec<f64> {
    let n = close.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let mut o = Vec::with_capacity(n);
    let mut c = Vec::with_capacity(n);
    let mut h = Vec::with_capacity(n);
    let mut l = Vec::with_capacity(n);
    for i in 0..n {
        o.push(safe_log(open[i]));
        c.push(safe_log(close[i]));
        h.push(safe_log(high[i]));
        l.push(safe_log(low[i]));
    }

    let mut o_ret = vec![f64::NAN; n];
    let mut c_ret = vec![f64::NAN; n];
    let mut rs = vec![f64::NAN; n];

    for i in 1..n {
        o_ret[i] = o[i] - c[i - 1];
        c_ret[i] = c[i] - o[i];
        let ho = h[i] - o[i];
        let hc = h[i] - c[i];
        let lo = l[i] - o[i];
        let lc = l[i] - c[i];
        rs[i] = (ho * hc) + (lo * lc);
    }

    let k = if window > 1 {
        0.34 / (1.34 + (window as f64 + 1.0) / (window as f64 - 1.0))
    } else {
        0.0
    };

    let sigma_o2 = rolling_var(&o_ret, window);
    let sigma_c2 = rolling_var(&c_ret, window);
    let sigma_rs2 = rolling_mean(&rs, window);

    let mut sigma = vec![0.0; n];
    for i in 0..n {
        let mut val = sigma_o2[i] + k * sigma_c2[i] + (1.0 - k) * sigma_rs2[i];
        if !val.is_finite() {
            val = 0.0;
        }
        sigma[i] = val.max(0.0).sqrt();
    }
    sigma
}

fn vol_ewma(close: &[f64], window: usize, lam: f64) -> Vec<f64> {
    let n = close.len();
    if n < 2 {
        return vec![0.0; n];
    }
    let lam = if lam <= 0.0 || lam >= 1.0 { 0.94 } else { lam };
    let mut r = Vec::with_capacity(n - 1);
    for i in 1..n {
        r.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    let mut var = vec![f64::NAN; n];
    let init = if window <= 1 {
        mean(&r.iter().map(|v| v * v).collect::<Vec<_>>())
    } else {
        let span = window.min(r.len());
        mean(&r[..span].iter().map(|v| v * v).collect::<Vec<_>>())
    };
    if n > 1 {
        var[1] = if init.is_finite() { init } else { 0.0 };
    }
    for i in 1..r.len() {
        let prev = if var[i].is_finite() { var[i] } else { init };
        var[i + 1] = (lam * prev) + ((1.0 - lam) * (r[i] * r[i]));
    }
    let mut sigma = vec![0.0; n];
    let mut last = 0.0;
    for i in 0..n {
        let val = var[i];
        if val.is_finite() && val > 0.0 {
            last = val.sqrt();
            sigma[i] = last;
        } else {
            sigma[i] = last;
        }
    }
    sigma
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn median_ignore_nan(values: &[f64]) -> f64 {
    let mut vals: Vec<f64> = values.iter().cloned().filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vals.len() / 2;
    if vals.len() % 2 == 0 {
        (vals[mid - 1] + vals[mid]) / 2.0
    } else {
        vals[mid]
    }
}

pub fn estimate_volatility(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    window: usize,
    method: &str,
    weights: Option<VolEnsembleWeights>,
    ewma_lambda: f64,
) -> Vec<f64> {
    let method = method.to_lowercase();
    if method == "ensemble" || method == "mix" || method == "blend" {
        let v_pk = vol_parkinson(high, low);
        let sigma_pk = rolling_mean(&v_pk, window).into_iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>();
        let v_gk = vol_garman_klass(open, high, low, close);
        let sigma_gk = rolling_mean(&v_gk, window).into_iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>();
        let v_rs = vol_rogers_satchell(open, high, low, close);
        let sigma_rs = rolling_mean(&v_rs, window).into_iter().map(|v| v.max(0.0).sqrt()).collect::<Vec<_>>();
        let sigma_yz = vol_yang_zhang(open, high, low, close, window);

        let mut out = vec![0.0; close.len()];
        let weights = weights.and_then(|w| w.normalize());
        for i in 0..out.len() {
            let stacked = [sigma_yz[i], sigma_gk[i], sigma_rs[i], sigma_pk[i]];
            if let Some(w) = weights {
                out[i] = stacked[0] * w[0] + stacked[1] * w[1] + stacked[2] * w[2] + stacked[3] * w[3];
            } else {
                let med = median_ignore_nan(&stacked);
                out[i] = if med.is_finite() { med } else { 0.0 };
            }
        }
        return out;
    }
    if method == "parkinson" || method == "park" {
        let v = vol_parkinson(high, low);
        return rolling_mean(&v, window).into_iter().map(|v| v.max(0.0).sqrt()).collect();
    }
    if method == "garman_klass" || method == "gk" {
        let v = vol_garman_klass(open, high, low, close);
        return rolling_mean(&v, window).into_iter().map(|v| v.max(0.0).sqrt()).collect();
    }
    if method == "rogers_satchell" || method == "rs" {
        let v = vol_rogers_satchell(open, high, low, close);
        return rolling_mean(&v, window).into_iter().map(|v| v.max(0.0).sqrt()).collect();
    }
    if method == "ewma" || method == "riskmetrics" {
        return vol_ewma(close, window, ewma_lambda);
    }
    vol_yang_zhang(open, high, low, close, window)
}

pub fn estimate_expected_shortfall(close: &[f64], window: usize, alpha: f64) -> Option<f64> {
    if window <= 2 || close.len() <= 2 {
        return None;
    }
    let mut r = Vec::with_capacity(close.len() - 1);
    for i in 1..close.len() {
        r.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    if r.len() < window {
        return None;
    }
    let tail = &r[r.len() - window..];
    let mut sorted = tail.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q_idx = ((1.0 - alpha).clamp(0.0, 1.0) * (sorted.len() as f64 - 1.0)).round() as usize;
    let q = sorted[q_idx];
    let losses: Vec<f64> = tail.iter().cloned().filter(|v| *v <= q).collect();
    if losses.is_empty() {
        return None;
    }
    Some(losses.iter().map(|v| v.abs()).sum::<f64>() / losses.len() as f64)
}

pub fn estimate_expected_shortfall_series(
    close: &[f64],
    window: usize,
    alpha: f64,
    step: usize,
    max_bars: usize,
) -> Option<Vec<f64>> {
    if window <= 2 || close.len() <= 2 || close.len() > max_bars {
        return None;
    }
    let mut r = Vec::with_capacity(close.len() - 1);
    for i in 1..close.len() {
        r.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    if r.len() < window {
        return None;
    }
    let mut es = vec![f64::NAN; r.len()];
    let step = step.max(1);
    let mut i = window - 1;
    while i < r.len() {
        let win = &r[i + 1 - window..=i];
        let mut sorted = win.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let q_idx = ((1.0 - alpha).clamp(0.0, 1.0) * (sorted.len() as f64 - 1.0)).round() as usize;
        let q = sorted[q_idx];
        let losses: Vec<f64> = win.iter().cloned().filter(|v| *v <= q).collect();
        if !losses.is_empty() {
            es[i] = losses.iter().map(|v| v.abs()).sum::<f64>() / losses.len() as f64;
        }
        i += step;
    }
    let mut last = 0.0;
    for v in &mut es {
        if v.is_finite() {
            last = *v;
        } else {
            *v = last;
        }
    }
    let mut out = Vec::with_capacity(close.len());
    out.push(f64::NAN);
    out.extend(es);
    Some(out)
}

pub fn estimate_hurst(close: &[f64], window: usize, max_lag: usize) -> Option<f64> {
    if window <= 10 || close.len() <= window {
        return None;
    }
    let start = close.len().saturating_sub(window + 1);
    let mut series = Vec::with_capacity(window);
    for i in (start + 1)..close.len() {
        series.push(safe_log(close[i]) - safe_log(close[i - 1]));
    }
    if series.len() < 20 {
        return None;
    }
    let max_lag = max_lag.min(series.len() / 2).max(2);
    let mut lags = Vec::new();
    let mut tau = Vec::new();
    for lag in 2..=max_lag {
        let mut diffs = Vec::with_capacity(series.len() - lag);
        for i in lag..series.len() {
            diffs.push(series[i] - series[i - lag]);
        }
        let std = stddev(&diffs);
        if std > 0.0 {
            lags.push(lag as f64);
            tau.push(std);
        }
    }
    if lags.len() < 2 {
        return None;
    }
    let log_lags: Vec<f64> = lags.iter().map(|v| v.ln()).collect();
    let log_tau: Vec<f64> = tau.iter().map(|v| v.ln()).collect();
    let slope = linreg_slope(&log_lags, &log_tau)?;
    Some(slope)
}

fn linreg_slope(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() != y.len() || x.len() < 2 {
        return None;
    }
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xy: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
    let sum_xx: f64 = x.iter().map(|v| v * v).sum();
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-12 {
        return None;
    }
    Some((n * sum_xy - sum_x * sum_y) / denom)
}

fn stddev(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mean = mean(values);
    let mut sum = 0.0;
    for v in values {
        let d = *v - mean;
        sum += d * d;
    }
    (sum / (values.len() as f64 - 1.0)).sqrt()
}

fn compute_ema(values: &[f64], period: usize) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let alpha = 2.0 / (period as f64 + 1.0);
    let mut out = Vec::with_capacity(values.len());
    let mut prev = values[0];
    out.push(prev);
    for i in 1..values.len() {
        prev = alpha * values[i] + (1.0 - alpha) * prev;
        out.push(prev);
    }
    out
}

fn compute_atr(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Vec<f64> {
    let n = close.len();
    let mut tr = vec![0.0; n];
    for i in 1..n {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        tr[i] = tr1.max(tr2).max(tr3);
    }
    compute_ema(&tr, period)
}

fn compute_adx(high: &[f64], low: &[f64], close: &[f64], period: usize) -> Option<f64> {
    let n = close.len();
    if n <= period + 1 {
        return None;
    }
    let mut tr = vec![0.0; n];
    let mut plus_dm = vec![0.0; n];
    let mut minus_dm = vec![0.0; n];
    for i in 1..n {
        let tr1 = high[i] - low[i];
        let tr2 = (high[i] - close[i - 1]).abs();
        let tr3 = (low[i] - close[i - 1]).abs();
        tr[i] = tr1.max(tr2).max(tr3);

        let up_move = high[i] - high[i - 1];
        let down_move = low[i - 1] - low[i];
        if up_move > down_move && up_move > 0.0 {
            plus_dm[i] = up_move;
        }
        if down_move > up_move && down_move > 0.0 {
            minus_dm[i] = down_move;
        }
    }

    let mut tr_sum: f64 = tr[1..=period].iter().sum();
    let mut plus_sum: f64 = plus_dm[1..=period].iter().sum();
    let mut minus_sum: f64 = minus_dm[1..=period].iter().sum();

    let mut dx = Vec::new();
    for i in (period + 1)..n {
        tr_sum = tr_sum - (tr_sum / period as f64) + tr[i];
        plus_sum = plus_sum - (plus_sum / period as f64) + plus_dm[i];
        minus_sum = minus_sum - (minus_sum / period as f64) + minus_dm[i];

        let plus_di = if tr_sum > 0.0 { 100.0 * plus_sum / tr_sum } else { 0.0 };
        let minus_di = if tr_sum > 0.0 { 100.0 * minus_sum / tr_sum } else { 0.0 };
        let denom = (plus_di + minus_di).max(1e-9);
        dx.push(100.0 * (plus_di - minus_di).abs() / denom);
    }

    if dx.len() < period {
        return None;
    }

    let mut adx = mean(&dx[..period]);
    for i in period..dx.len() {
        adx = ((adx * (period as f64 - 1.0)) + dx[i]) / period as f64;
    }
    Some(adx)
}

pub fn infer_regime(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
) -> String {
    let adx_val = compute_adx(high, low, close, settings.atr_period);
    let hurst_val = estimate_hurst(close, settings.hurst_window, 20);

    if let (Some(adx), Some(hurst)) = (adx_val, hurst_val) {
        if adx >= settings.regime_adx_trend && hurst >= settings.hurst_trend {
            return "trend".to_string();
        }
        if adx <= settings.regime_adx_range && hurst <= settings.hurst_range {
            return "range".to_string();
        }
    }

    if let Some(adx) = adx_val {
        if adx >= settings.regime_adx_trend {
            return "trend".to_string();
        }
        if adx <= settings.regime_adx_range {
            return "range".to_string();
        }
    }

    if let Some(hurst) = hurst_val {
        if hurst >= settings.hurst_trend {
            return "trend".to_string();
        }
        if hurst <= settings.hurst_range {
            return "range".to_string();
        }
    }

    let ema_fast = compute_ema(close, settings.ema_fast_period);
    let ema_slow = compute_ema(close, settings.ema_slow_period);
    let atr = compute_atr(high, low, close, settings.atr_period);
    if let (Some(ef), Some(es), Some(a)) = (ema_fast.last(), ema_slow.last(), atr.last()) {
        if *a > 0.0 {
            let spread = (ef - es).abs();
            let strength = spread / a;
            if strength >= 0.6 {
                return "trend".to_string();
            }
            if strength <= 0.3 {
                return "range".to_string();
            }
        }
    }

    "neutral".to_string()
}

pub fn compute_stop_distance_series(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
) -> Option<Vec<f64>> {
    if close.len() < settings.vol_window.max(settings.tail_window).max(5) {
        return None;
    }

    let regime = infer_regime(open, high, low, close, settings);
    let weights = match regime.as_str() {
        "trend" => settings.weights_trend.or(settings.weights),
        "range" => settings.weights_range.or(settings.weights),
        _ => settings.weights,
    };

    let sigma = estimate_volatility(
        open,
        high,
        low,
        close,
        settings.vol_window,
        &settings.vol_estimator,
        weights,
        settings.ewma_lambda,
    );
    let scale = (settings.vol_horizon_bars.max(1) as f64).sqrt();
    let vol_dist: Vec<f64> = close
        .iter()
        .zip(sigma.iter())
        .map(|(c, s)| c * s * scale)
        .collect();

    let es_series = estimate_expected_shortfall_series(
        close,
        settings.tail_window,
        settings.tail_alpha,
        settings.tail_step,
        settings.tail_max_bars,
    );
    let tail_dist = if let Some(es) = es_series {
        close
            .iter()
            .zip(es.iter())
            .map(|(c, s)| c * s * scale)
            .collect::<Vec<_>>()
    } else {
        vec![0.0; close.len()]
    };

    let mut dist = Vec::with_capacity(close.len());
    for i in 0..close.len() {
        let base = (settings.stop_k_vol * vol_dist[i]).max(settings.stop_k_tail * tail_dist[i]);
        dist.push(base.max(settings.meta_label_min_dist));
    }

    let med = median_ignore_nan(&dist);
    if !med.is_finite() || med <= 0.0 {
        return None;
    }
    for v in &mut dist {
        if !v.is_finite() {
            *v = med;
        }
    }

    Some(dist)
}

pub fn infer_stop_target_pips(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
    settings: &StopTargetSettings,
    pip_size: f64,
) -> Option<(f64, f64, f64)> {
    if close.len() < settings.vol_window.max(settings.tail_window).max(5) {
        return None;
    }

    let regime = infer_regime(open, high, low, close, settings);
    let weights = match regime.as_str() {
        "trend" => settings.weights_trend.or(settings.weights),
        "range" => settings.weights_range.or(settings.weights),
        _ => settings.weights,
    };

    let sigma = estimate_volatility(
        open,
        high,
        low,
        close,
        settings.vol_window,
        &settings.vol_estimator,
        weights,
        settings.ewma_lambda,
    );
    let sigma_last = *sigma.last().unwrap_or(&0.0);
    let es = estimate_expected_shortfall(close, settings.tail_window, settings.tail_alpha).unwrap_or(0.0);
    let price = *close.last().unwrap_or(&0.0);
    let scale = (settings.vol_horizon_bars.max(1) as f64).sqrt();
    let vol_dist = price * sigma_last * scale;
    let tail_dist = price * es * scale;
    let dist = (settings.stop_k_vol * vol_dist)
        .max(settings.stop_k_tail * tail_dist)
        .max(settings.meta_label_min_dist);
    if !dist.is_finite() || dist <= 0.0 {
        return None;
    }

    let sl_pips = dist / pip_size.max(1e-9);
    let rr = match regime.as_str() {
        "trend" => settings.rr_trend,
        "range" => settings.rr_range,
        _ => settings.rr_neutral,
    };
    let tp_pips = sl_pips * rr;
    if !tp_pips.is_finite() || tp_pips <= 0.0 {
        return None;
    }
    Some((sl_pips, tp_pips, rr))
}
