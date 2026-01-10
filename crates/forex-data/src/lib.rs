use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use talib::common::{ta_initialize, TimePeriodKwargs};
use talib::momentum::{ta_adx, ta_cci, ta_macd, ta_rsi, MacdKwargs};
use talib::overlap::{ta_bbands, ta_ema, ta_sma, BBANDSKwargs};
use talib::volatility::{ta_atr, ta_natr, ATRKwargs, NATRKwargs};
use talib_sys::{TA_MAType, TA_MAType_TA_MAType_SMA};

#[derive(Debug, Clone)]
pub struct Ohlcv {
    pub timestamp: Option<Vec<i64>>,
    pub open: Vec<f64>,
    pub high: Vec<f64>,
    pub low: Vec<f64>,
    pub close: Vec<f64>,
    pub volume: Option<Vec<f64>>,
}

impl Ohlcv {
    pub fn len(&self) -> usize {
        self.close.len()
    }

    pub fn is_empty(&self) -> bool {
        self.close.is_empty()
    }
}

fn is_sorted_timestamps(ts: &[i64]) -> bool {
    ts.windows(2).all(|w| w[0] <= w[1])
}

fn sort_ohlcv_by_timestamp(ohlcv: &Ohlcv) -> Ohlcv {
    let Some(ts) = &ohlcv.timestamp else {
        return ohlcv.clone();
    };
    if ts.len() != ohlcv.len() || is_sorted_timestamps(ts) {
        return ohlcv.clone();
    }
    let mut idx: Vec<usize> = (0..ts.len()).collect();
    idx.sort_by_key(|&i| ts[i]);

    let reorder = |src: &Vec<f64>| idx.iter().map(|&i| src[i]).collect::<Vec<f64>>();
    let sorted_ts = idx.iter().map(|&i| ts[i]).collect::<Vec<i64>>();
    let volume = ohlcv
        .volume
        .as_ref()
        .map(|v| idx.iter().map(|&i| v[i]).collect::<Vec<f64>>());

    Ohlcv {
        timestamp: Some(sorted_ts),
        open: reorder(&ohlcv.open),
        high: reorder(&ohlcv.high),
        low: reorder(&ohlcv.low),
        close: reorder(&ohlcv.close),
        volume,
    }
}

fn pad_vec(mut values: Vec<f64>, len: usize) -> Vec<f64> {
    if values.len() < len {
        values.resize(len, f64::NAN);
    } else if values.len() > len {
        values.truncate(len);
    }
    values
}

static TALIB_INIT: OnceLock<Result<(), anyhow::Error>> = OnceLock::new();

fn ensure_talib_init() -> Result<()> {
    let init = TALIB_INIT.get_or_init(|| {
        ta_initialize().map_err(|e| anyhow::anyhow!("TA-Lib init failed: {:?}", e))
    });
    match init {
        Ok(_) => Ok(()),
        Err(err) => Err(anyhow::anyhow!(err.to_string())),
    }
}

fn compute_talib_indicators(ohlcv: &Ohlcv) -> Result<Vec<(String, Vec<f64>)>> {
    if ohlcv.is_empty() {
        bail!("empty OHLCV data");
    }
    ensure_talib_init()?;

    let len = ohlcv.len();
    let close_ptr = ohlcv.close.as_ptr();
    let high_ptr = ohlcv.high.as_ptr();
    let low_ptr = ohlcv.low.as_ptr();

    let mut out: Vec<(String, Vec<f64>)> = Vec::new();

    let period14 = TimePeriodKwargs { timeperiod: 14 };
    let period20 = TimePeriodKwargs { timeperiod: 20 };

    if let Ok(values) = ta_rsi(close_ptr, len, &period14) {
        out.push(("rsi_14".to_string(), pad_vec(values, len)));
    }
    if let Ok(values) = ta_adx(high_ptr, low_ptr, close_ptr, len, &period14) {
        out.push(("adx_14".to_string(), pad_vec(values, len)));
    }
    if let Ok(values) = ta_cci(high_ptr, low_ptr, close_ptr, len, &period20) {
        out.push(("cci_20".to_string(), pad_vec(values, len)));
    }

    let macd_kwargs = MacdKwargs {
        fastperiod: 12,
        slowperiod: 26,
        signalperiod: 9,
    };
    if let Ok((macd, signal, hist)) = ta_macd(close_ptr, len, &macd_kwargs) {
        out.push(("macd".to_string(), pad_vec(macd, len)));
        out.push(("macd_signal".to_string(), pad_vec(signal, len)));
        out.push(("macd_hist".to_string(), pad_vec(hist, len)));
    }

    let atr_kwargs = ATRKwargs { timeperiod: 14 };
    if let Ok(values) = ta_atr(high_ptr, low_ptr, close_ptr, len, &atr_kwargs) {
        out.push(("atr_14".to_string(), pad_vec(values, len)));
    }
    let natr_kwargs = NATRKwargs { timeperiod: 14 };
    if let Ok(values) = ta_natr(high_ptr, low_ptr, close_ptr, len, &natr_kwargs) {
        out.push(("natr_14".to_string(), pad_vec(values, len)));
    }

    if let Ok(values) = ta_sma(close_ptr, len, &period20) {
        out.push(("sma_20".to_string(), pad_vec(values, len)));
    }
    if let Ok(values) = ta_ema(close_ptr, len, &period20) {
        out.push(("ema_20".to_string(), pad_vec(values, len)));
    }

    let bb_kwargs = BBANDSKwargs {
        timeperiod: 20,
        nbdevup: 2.0,
        nbdevdn: 2.0,
        matype: TA_MAType_TA_MAType_SMA as TA_MAType,
    };
    if let Ok((upper, middle, lower)) = ta_bbands(close_ptr, len, &bb_kwargs) {
        let upper = pad_vec(upper, len);
        let middle = pad_vec(middle, len);
        let lower = pad_vec(lower, len);
        out.push(("bb_upper".to_string(), upper.clone()));
        out.push(("bb_middle".to_string(), middle.clone()));
        out.push(("bb_lower".to_string(), lower.clone()));
        let mut width = Vec::with_capacity(len);
        for i in 0..len {
            let mid = middle.get(i).copied().unwrap_or(f64::NAN);
            let up = upper.get(i).copied().unwrap_or(f64::NAN);
            let lo = lower.get(i).copied().unwrap_or(f64::NAN);
            if !mid.is_finite() || mid.abs() <= f64::EPSILON {
                width.push(f64::NAN);
            } else {
                width.push((up - lo) / mid.abs());
            }
        }
        out.push(("bb_width".to_string(), width));
    }

    if out.is_empty() {
        bail!("TA-Lib produced no indicators");
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct FeatureMatrix {
    pub data: Array2<f32>,
    pub names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct FeatureFrame {
    pub timestamps: Vec<i64>,
    pub names: Vec<String>,
    pub data: Array2<f32>,
}

#[derive(Debug, Clone)]
pub struct SymbolDataset {
    pub symbol: String,
    pub frames: HashMap<String, Ohlcv>,
}

#[derive(Debug, Clone)]
pub struct FeatureCache {
    pub dir: PathBuf,
    pub ttl_minutes: u64,
    pub enabled: bool,
}

impl FeatureCache {
    pub fn new(dir: impl AsRef<Path>, ttl_minutes: u64, enabled: bool) -> Self {
        Self {
            dir: dir.as_ref().to_path_buf(),
            ttl_minutes,
            enabled,
        }
    }

    fn is_fresh(&self, path: &Path) -> bool {
        if !self.enabled {
            return false;
        }
        if self.ttl_minutes == 0 {
            return true;
        }
        let Ok(metadata) = std::fs::metadata(path) else {
            return false;
        };
        let Ok(modified) = metadata.modified() else {
            return false;
        };
        let Ok(elapsed) = modified.elapsed() else {
            return false;
        };
        elapsed.as_secs() <= self.ttl_minutes * 60
    }

    pub fn load(&self, key: &str) -> Result<Option<FeatureFrame>> {
        if !self.enabled {
            return Ok(None);
        }
        let mut path = self.dir.clone();
        path.push(format!("{key}.parquet"));
        if !path.exists() {
            return Ok(None);
        }
        if !self.is_fresh(&path) {
            return Ok(None);
        }
        let file = std::fs::File::open(&path)?;
        let df = ParquetReader::new(file).finish()?;
        Ok(Some(df_to_feature_frame(&df)?))
    }

    pub fn store(&self, key: &str, frame: &FeatureFrame) -> Result<()> {
        if !self.enabled {
            return Ok(());
        }
        std::fs::create_dir_all(&self.dir)?;
        let mut path = self.dir.clone();
        path.push(format!("{key}.parquet"));
        let mut df = feature_frame_to_df(frame)?;
        let file = std::fs::File::create(&path)?;
        ParquetWriter::new(file).finish(&mut df)?;
        Ok(())
    }
}

impl SymbolDataset {
    pub fn timeframe(&self, tf: &str) -> Option<&Ohlcv> {
        self.frames.get(tf)
    }

    pub fn timeframes(&self) -> Vec<String> {
        let mut out: Vec<String> = self.frames.keys().cloned().collect();
        out.sort();
        out
    }
}

pub const MANDATORY_TFS: [&str; 11] = [
    "M1", "M3", "M5", "M15", "M30", "H1", "H2", "H4", "D1", "W1", "MN1",
];

fn series_to_f64(series: &Series) -> Result<Vec<f64>> {
    let casted = series.cast(&DataType::Float64)?;
    let chunked = casted.f64().context("series cast to f64 failed")?;
    Ok(chunked.into_iter().map(|v| v.unwrap_or(0.0)).collect())
}

fn series_to_f32(series: &Series, n_rows: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n_rows];
    let casted = series
        .cast(&DataType::Float64)
        .unwrap_or_else(|_| series.clone());
    if let Ok(chunked) = casted.f64() {
        for (i, v) in chunked.into_iter().enumerate().take(n_rows) {
            out[i] = v.unwrap_or(0.0) as f32;
        }
    }
    out
}

fn find_series(df: &DataFrame, candidates: &[&str]) -> Option<Series> {
    for name in df.get_column_names() {
        let lower = name.to_ascii_lowercase();
        if candidates.iter().any(|c| lower == *c) {
            // In polars 0.47, column() returns &Column, convert to Series
            return df.column(name).ok().map(|col| col.as_materialized_series().clone());
        }
    }
    None
}

fn extract_timestamps(df: &DataFrame) -> Result<Option<Vec<i64>>> {
    let series = match find_series(df, &["timestamp", "time", "datetime", "date"]) {
        Some(s) => s,
        None => return Ok(None),
    };
    let casted = series.cast(&DataType::Int64)?;
    let chunked = casted.i64().context("timestamp cast to i64 failed")?;
    Ok(Some(chunked.into_iter().map(|v| v.unwrap_or(0)).collect()))
}

fn feature_frame_to_df(frame: &FeatureFrame) -> Result<DataFrame> {
    let mut cols: Vec<Column> = Vec::with_capacity(frame.names.len() + 1);
    cols.push(Series::new("timestamp".into(), frame.timestamps.clone()).into());
    for (idx, name) in frame.names.iter().enumerate() {
        let mut col = Vec::with_capacity(frame.data.nrows());
        for row in 0..frame.data.nrows() {
            col.push(frame.data[(row, idx)]);
        }
        cols.push(Series::new(name.as_str().into(), col).into());
    }
    Ok(DataFrame::new(cols)?)
}

fn df_to_feature_frame(df: &DataFrame) -> Result<FeatureFrame> {
    let timestamps = extract_timestamps(df)?.context("cached features missing timestamp column")?;
    let mut names = Vec::new();
    let mut columns: Vec<Vec<f32>> = Vec::new();
    for col in df.get_columns() {
        let series = col.as_materialized_series();
        if series.name().eq_ignore_ascii_case("timestamp") {
            continue;
        }
        names.push(series.name().to_string());
        columns.push(series_to_f32(series, timestamps.len()));
    }
    let n_rows = timestamps.len();
    let n_cols = columns.len();
    let mut data = Array2::<f32>::zeros((n_rows, n_cols));
    for (col_idx, vals) in columns.iter().enumerate() {
        let len = vals.len().min(n_rows);
        for i in 0..len {
            data[(i, col_idx)] = vals[i];
        }
    }
    Ok(FeatureFrame {
        timestamps,
        names,
        data,
    })
}

pub fn load_parquet(path: impl AsRef<Path>) -> Result<Ohlcv> {
    let path = path.as_ref();
    let file = std::fs::File::open(path)
        .with_context(|| format!("failed to open parquet file: {}", path.display()))?;
    let df = ParquetReader::new(file).finish()?;

    let timestamp = extract_timestamps(&df)?;
    let open = find_series(&df, &["open", "o"]).context("missing open column")?;
    let high = find_series(&df, &["high", "h"]).context("missing high column")?;
    let low = find_series(&df, &["low", "l"]).context("missing low column")?;
    let close = find_series(&df, &["close", "c"]).context("missing close column")?;
    let volume = find_series(&df, &["volume", "vol", "v"]);

    let open = series_to_f64(&open)?;
    let high = series_to_f64(&high)?;
    let low = series_to_f64(&low)?;
    let close = series_to_f64(&close)?;
    let volume = match volume {
        Some(ref series) => Some(series_to_f64(series)?),
        None => None,
    };

    let n = close.len();
    if open.len() != n || high.len() != n || low.len() != n {
        bail!("OHLC columns have mismatched lengths");
    }
    if let Some(ref vol) = volume {
        if vol.len() != n {
            bail!("volume column length does not match OHLC length");
        }
    }
    if let Some(ref ts) = timestamp {
        if ts.len() != n {
            bail!("timestamp column length does not match OHLC length");
        }
    }

    Ok(Ohlcv {
        timestamp,
        open,
        high,
        low,
        close,
        volume,
    })
}

pub fn load_symbol_dataset(root: impl AsRef<Path>, symbol: &str) -> Result<SymbolDataset> {
    let tfs = discover_timeframes(&root, symbol)?;
    if tfs.is_empty() {
        bail!("no timeframes discovered for symbol={}", symbol);
    }
    let mut frames = HashMap::new();
    for tf in tfs {
        let ohlcv = load_symbol_timeframe(&root, symbol, &tf)?;
        frames.insert(tf, ohlcv);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
    })
}

pub fn load_symbol_dataset_with_timeframes(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframes: &[&str],
) -> Result<SymbolDataset> {
    let mut frames = HashMap::new();
    for tf in timeframes {
        let ohlcv = load_symbol_timeframe(&root, symbol, tf)?;
        frames.insert(tf.to_string(), ohlcv);
    }
    Ok(SymbolDataset {
        symbol: symbol.to_string(),
        frames,
    })
}

pub fn compute_talib_features(ohlcv: &Ohlcv) -> Result<FeatureMatrix> {
    let sorted = sort_ohlcv_by_timestamp(ohlcv);
    let n_rows = sorted.len();
    if n_rows == 0 {
        bail!("empty OHLCV data");
    }

    let indicators = compute_talib_indicators(&sorted)?;
    let mut names = Vec::with_capacity(indicators.len());
    let mut columns: Vec<Vec<f32>> = Vec::with_capacity(indicators.len());

    for (name, values) in indicators {
        names.push(format!("ta_{name}"));
        let vals = pad_vec(values, n_rows);
        columns.push(vals.iter().map(|v| *v as f32).collect());
    }

    let n_cols = columns.len();
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    for (col_idx, vals) in columns.iter().enumerate() {
        let len = vals.len().min(n_rows);
        for i in 0..len {
            out[(i, col_idx)] = vals[i];
        }
    }

    Ok(FeatureMatrix { data: out, names })
}

pub fn compute_talib_feature_frame(ohlcv: &Ohlcv, include_raw: bool) -> Result<FeatureFrame> {
    let sorted = sort_ohlcv_by_timestamp(ohlcv);
    let n_rows = sorted.len();
    if n_rows == 0 {
        bail!("empty OHLCV data");
    }

    let timestamps = sorted
        .timestamp
        .clone()
        .unwrap_or_else(|| (0..n_rows as i64).collect());

    let indicators = compute_talib_indicators(&sorted)?;
    let mut names = Vec::new();
    let mut columns: Vec<Vec<f32>> = Vec::new();

    if include_raw {
        names.push("open".to_string());
        columns.push(sorted.open.iter().map(|v| *v as f32).collect());
        names.push("high".to_string());
        columns.push(sorted.high.iter().map(|v| *v as f32).collect());
        names.push("low".to_string());
        columns.push(sorted.low.iter().map(|v| *v as f32).collect());
        names.push("close".to_string());
        columns.push(sorted.close.iter().map(|v| *v as f32).collect());
        if let Some(volume) = &sorted.volume {
            names.push("volume".to_string());
            columns.push(volume.iter().map(|v| *v as f32).collect());
        }
    }

    for (name, values) in indicators {
        names.push(format!("ta_{name}"));
        let vals = pad_vec(values, n_rows);
        columns.push(vals.iter().map(|v| *v as f32).collect());
    }

    let n_cols = columns.len();
    let mut out = Array2::<f32>::zeros((n_rows, n_cols));
    for (col_idx, vals) in columns.iter().enumerate() {
        let len = vals.len().min(n_rows);
        for i in 0..len {
            out[(i, col_idx)] = vals[i];
        }
    }

    Ok(FeatureFrame {
        timestamps,
        names,
        data: out,
    })
}

fn select_htf_indices(names: &[String]) -> Vec<usize> {
    let patterns = ["rsi", "macd", "atr", "bb", "bb_width"];
    names
        .iter()
        .enumerate()
        .filter_map(|(idx, name)| {
            let lower = name.to_ascii_lowercase();
            if patterns.iter().any(|p| lower.contains(p)) {
                Some(idx)
            } else {
                None
            }
        })
        .collect()
}

fn select_columns(data: &Array2<f32>, indices: &[usize]) -> Array2<f32> {
    let n_rows = data.nrows();
    let mut out = Array2::<f32>::zeros((n_rows, indices.len()));
    for (col_pos, col_idx) in indices.iter().enumerate() {
        for row in 0..n_rows {
            out[(row, col_pos)] = data[(row, *col_idx)];
        }
    }
    out
}

fn align_features(base_ts: &[i64], htf_ts: &[i64], htf_data: &Array2<f32>) -> Array2<f32> {
    let n_base = base_ts.len();
    let n_htf = htf_ts.len();
    let n_cols = htf_data.ncols();
    let mut out = Array2::<f32>::zeros((n_base, n_cols));
    if n_htf == 0 || n_base == 0 {
        return out;
    }
    let mut j = 0usize;
    for i in 0..n_base {
        let target = base_ts[i];
        while j + 1 < n_htf && htf_ts[j + 1] <= target {
            j += 1;
        }
        if htf_ts[j] > target {
            continue;
        }
        if j == 0 {
            continue;
        }
        let src = j - 1;
        for c in 0..n_cols {
            out[(i, c)] = htf_data[(src, c)];
        }
    }
    out
}

fn hstack(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (rows, cols_a) = a.dim();
    let cols_b = b.ncols();
    let mut out = Array2::<f32>::zeros((rows, cols_a + cols_b));
    for r in 0..rows {
        for c in 0..cols_a {
            out[(r, c)] = a[(r, c)];
        }
        for c in 0..cols_b {
            out[(r, cols_a + c)] = b[(r, c)];
        }
    }
    out
}

pub fn missing_timeframes(dataset: &SymbolDataset, required: &[&str]) -> Vec<String> {
    let mut missing = Vec::new();
    for tf in required {
        if !dataset.frames.contains_key(*tf) {
            missing.push((*tf).to_string());
        }
    }
    missing
}

pub fn ensure_timeframes(dataset: &SymbolDataset, required: &[&str]) -> Result<()> {
    let missing = missing_timeframes(dataset, required);
    if !missing.is_empty() {
        bail!("missing timeframes: {}", missing.join(", "));
    }
    Ok(())
}

fn timeframe_to_ms(tf: &str) -> Option<i64> {
    match tf.to_ascii_uppercase().as_str() {
        "M1" => Some(60_000),
        "M3" => Some(180_000),
        "M5" => Some(300_000),
        "M15" => Some(900_000),
        "M30" => Some(1_800_000),
        "H1" => Some(3_600_000),
        "H2" => Some(7_200_000),
        "H4" => Some(14_400_000),
        "D1" => Some(86_400_000),
        "W1" => Some(604_800_000),
        "MN1" => Some(2_592_000_000),
        _ => None,
    }
}

pub fn resample_ohlcv(ohlcv: &Ohlcv, target_tf: &str) -> Result<Ohlcv> {
    let Some(ts) = ohlcv.timestamp.clone() else {
        bail!("cannot resample without timestamps");
    };
    let Some(bucket_ms) = timeframe_to_ms(target_tf) else {
        bail!("unsupported timeframe: {}", target_tf);
    };
    if ts.is_empty() {
        bail!("empty timestamp series");
    }

    let mut out_ts = Vec::new();
    let mut out_open = Vec::new();
    let mut out_high = Vec::new();
    let mut out_low = Vec::new();
    let mut out_close = Vec::new();
    let mut out_vol: Option<Vec<f64>> = ohlcv.volume.as_ref().map(|_| Vec::new());

    let mut current_bucket = ts[0] / bucket_ms;
    let mut open = ohlcv.open[0];
    let mut high = ohlcv.high[0];
    let mut low = ohlcv.low[0];
    let mut close = ohlcv.close[0];
    let mut volume = ohlcv.volume.as_ref().map(|v| v[0]).unwrap_or(0.0);

    for i in 1..ts.len() {
        let bucket = ts[i] / bucket_ms;
        if bucket != current_bucket {
            out_ts.push(ts[i - 1]);
            out_open.push(open);
            out_high.push(high);
            out_low.push(low);
            out_close.push(close);
            if let Some(ref mut vec) = out_vol {
                vec.push(volume);
            }

            current_bucket = bucket;
            open = ohlcv.open[i];
            high = ohlcv.high[i];
            low = ohlcv.low[i];
            close = ohlcv.close[i];
            volume = ohlcv.volume.as_ref().map(|v| v[i]).unwrap_or(0.0);
        } else {
            if ohlcv.high[i] > high {
                high = ohlcv.high[i];
            }
            if ohlcv.low[i] < low {
                low = ohlcv.low[i];
            }
            close = ohlcv.close[i];
            volume += ohlcv.volume.as_ref().map(|v| v[i]).unwrap_or(0.0);
        }
    }

    out_ts.push(*ts.last().unwrap());
    out_open.push(open);
    out_high.push(high);
    out_low.push(low);
    out_close.push(close);
    if let Some(ref mut vec) = out_vol {
        vec.push(volume);
    }

    Ok(Ohlcv {
        timestamp: Some(out_ts),
        open: out_open,
        high: out_high,
        low: out_low,
        close: out_close,
        volume: out_vol,
    })
}

pub fn ensure_timeframes_with_resample(
    dataset: &SymbolDataset,
    base_tf: &str,
    targets: &[&str],
) -> Result<SymbolDataset> {
    let base = dataset
        .frames
        .get(base_tf)
        .context("base timeframe missing for resample")?;
    let mut frames = dataset.frames.clone();
    for tf in targets {
        if frames.contains_key(*tf) {
            continue;
        }
        let resampled = resample_ohlcv(base, tf)?;
        frames.insert((*tf).to_string(), resampled);
    }
    Ok(SymbolDataset {
        symbol: dataset.symbol.clone(),
        frames,
    })
}

pub fn prepare_multitimeframe_features(
    dataset: &SymbolDataset,
    base_tf: &str,
    higher_tfs: &[&str],
    cache: Option<&FeatureCache>,
) -> Result<FeatureFrame> {
    let base_tf = if dataset.frames.contains_key(base_tf) {
        base_tf.to_string()
    } else if dataset.frames.contains_key("M5") {
        "M5".to_string()
    } else if dataset.frames.contains_key("M1") {
        "M1".to_string()
    } else {
        dataset
            .frames
            .keys()
            .next()
            .cloned()
            .context("no timeframes available")?
    };

    let base_ohlcv = dataset
        .frames
        .get(&base_tf)
        .context("base timeframe data missing")?;

    let base_key = format!("{}_{}_base", dataset.symbol, base_tf);
    let base_frame = if let Some(cache) = cache {
        if let Some(frame) = cache.load(&base_key)? {
            frame
        } else {
            let frame = compute_talib_feature_frame(base_ohlcv, true)?;
            cache.store(&base_key, &frame)?;
            frame
        }
    } else {
        compute_talib_feature_frame(base_ohlcv, true)?
    };

    let base_ts = base_frame.timestamps.clone();
    let mut names = base_frame.names.clone();
    let mut data = base_frame.data.clone();

    let mut targets: Vec<String> = if higher_tfs.is_empty() {
        dataset
            .frames
            .keys()
            .filter(|tf| *tf != &base_tf)
            .cloned()
            .collect()
    } else {
        higher_tfs.iter().map(|tf| tf.to_string()).collect()
    };
    targets.sort();

    for tf in targets {
        if tf == base_tf {
            continue;
        }
        let htf_ohlcv = match dataset.frames.get(&tf) {
            Some(val) => val,
            None => continue,
        };
        let htf_key = format!("{}_{}_htf", dataset.symbol, tf);
        let htf_frame = if let Some(cache) = cache {
            if let Some(frame) = cache.load(&htf_key)? {
                frame
            } else {
                let frame = compute_talib_feature_frame(htf_ohlcv, false)?;
                cache.store(&htf_key, &frame)?;
                frame
            }
        } else {
            compute_talib_feature_frame(htf_ohlcv, false)?
        };

        if htf_frame.timestamps.is_empty() {
            continue;
        }
        let indices = select_htf_indices(&htf_frame.names);
        if indices.is_empty() {
            continue;
        }
        let subset = select_columns(&htf_frame.data, &indices);
        let aligned = align_features(&base_ts, &htf_frame.timestamps, &subset);
        let prefixed_names: Vec<String> = indices
            .iter()
            .map(|idx| format!("{}_{}", tf, htf_frame.names[*idx]))
            .collect();

        data = hstack(&data, &aligned);
        names.extend(prefixed_names);
    }

    Ok(FeatureFrame {
        timestamps: base_ts,
        names,
        data,
    })
}

pub fn load_symbol_timeframe(
    root: impl AsRef<Path>,
    symbol: &str,
    timeframe: &str,
) -> Result<Ohlcv> {
    let mut path = PathBuf::from(root.as_ref());
    path.push(format!("symbol={}", symbol));
    path.push(format!("timeframe={}", timeframe));
    path.push("data.parquet");
    load_parquet(&path)
}

pub fn discover_symbols(root: impl AsRef<Path>) -> Result<Vec<String>> {
    let root = root.as_ref();
    let mut out = Vec::new();
    for entry in std::fs::read_dir(root)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(symbol) = name.strip_prefix("symbol=") {
            out.push(symbol.to_string());
        }
    }
    out.sort();
    Ok(out)
}

pub fn discover_timeframes(root: impl AsRef<Path>, symbol: &str) -> Result<Vec<String>> {
    let mut path = PathBuf::from(root.as_ref());
    path.push(format!("symbol={}", symbol));
    let mut out = Vec::new();
    for entry in std::fs::read_dir(path)? {
        let entry = entry?;
        if !entry.file_type()?.is_dir() {
            continue;
        }
        let name = entry.file_name().to_string_lossy().to_string();
        if let Some(tf) = name.strip_prefix("timeframe=") {
            out.push(tf.to_string());
        }
    }
    out.sort();
    Ok(out)
}
