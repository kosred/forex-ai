use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use ndarray::Array2;
use polars::prelude::*;
use ta_lib_in_rust::indicators::add_technical_indicators;

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
        let df = feature_frame_to_df(frame)?;
        let file = std::fs::File::create(&path)?;
        ParquetWriter::new(file).finish(&df)?;
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
    let casted = series.cast(&DataType::Float64).unwrap_or_else(|_| series.clone());
    if let Ok(chunked) = casted.f64() {
        for (i, v) in chunked.into_iter().enumerate().take(n_rows) {
            out[i] = v.unwrap_or(0.0) as f32;
        }
    }
    out
}

fn find_series<'a>(df: &'a DataFrame, candidates: &[&str]) -> Option<&'a Series> {
    for name in df.get_column_names() {
        let lower = name.to_ascii_lowercase();
        if candidates.iter().any(|c| lower == *c) {
            return df.column(name).ok();
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
    let mut cols: Vec<Series> = Vec::with_capacity(frame.names.len() + 1);
    cols.push(Series::new("timestamp", frame.timestamps.clone()));
    for (idx, name) in frame.names.iter().enumerate() {
        let mut col = Vec::with_capacity(frame.data.nrows());
        for row in 0..frame.data.nrows() {
            col.push(frame.data[(row, idx)]);
        }
        cols.push(Series::new(name.as_str(), col));
    }
    Ok(DataFrame::new(cols)?)
}

fn df_to_feature_frame(df: &DataFrame) -> Result<FeatureFrame> {
    let timestamps = extract_timestamps(df)?
        .context("cached features missing timestamp column")?;
    let mut names = Vec::new();
    let mut columns: Vec<Vec<f32>> = Vec::new();
    for series in df.get_columns() {
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
    let open = find_series(&df, &["open", "o"])
        .context("missing open column")?;
    let high = find_series(&df, &["high", "h"])
        .context("missing high column")?;
    let low = find_series(&df, &["low", "l"])
        .context("missing low column")?;
    let close = find_series(&df, &["close", "c"])
        .context("missing close column")?;
    let volume = find_series(&df, &["volume", "vol", "v"]);

    let open = series_to_f64(open)?;
    let high = series_to_f64(high)?;
    let low = series_to_f64(low)?;
    let close = series_to_f64(close)?;
    let volume = match volume {
        Some(series) => Some(series_to_f64(series)?),
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

pub fn load_symbol_dataset(
    root: impl AsRef<Path>,
    symbol: &str,
) -> Result<SymbolDataset> {
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
    let n_rows = ohlcv.len();
    if n_rows == 0 {
        bail!("empty OHLCV data");
    }

    let volume = ohlcv
        .volume
        .clone()
        .unwrap_or_else(|| vec![0.0_f64; n_rows]);

    let mut df = DataFrame::new(vec![
        Series::new("open", ohlcv.open.clone()),
        Series::new("high", ohlcv.high.clone()),
        Series::new("low", ohlcv.low.clone()),
        Series::new("close", ohlcv.close.clone()),
        Series::new("volume", volume),
    ])?;

    let original_cols: HashSet<String> = df
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();

    let df = add_technical_indicators(&mut df)
        .map_err(|e| anyhow::anyhow!("ta-lib-in-rust failed: {e}"))?;

    let mut names = Vec::new();
    let mut columns = Vec::new();
    for series in df.get_columns() {
        let name = series.name();
        if original_cols.contains(name) {
            continue;
        }
        let mut col_name = String::from("ta_");
        col_name.push_str(name);
        names.push(col_name);
        columns.push(series_to_f32(series, n_rows));
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
    let n_rows = ohlcv.len();
    if n_rows == 0 {
        bail!("empty OHLCV data");
    }

    let timestamps = ohlcv
        .timestamp
        .clone()
        .unwrap_or_else(|| (0..n_rows as i64).collect());
    let volume = ohlcv
        .volume
        .clone()
        .unwrap_or_else(|| vec![0.0_f64; n_rows]);

    let mut df = DataFrame::new(vec![
        Series::new("timestamp", timestamps),
        Series::new("open", ohlcv.open.clone()),
        Series::new("high", ohlcv.high.clone()),
        Series::new("low", ohlcv.low.clone()),
        Series::new("close", ohlcv.close.clone()),
        Series::new("volume", volume),
    ])?;

    df = df.sort(["timestamp"], SortMultipleOptions::default())?;

    let original_cols: HashSet<String> = df
        .get_column_names()
        .iter()
        .map(|name| name.to_string())
        .collect();

    let df = add_technical_indicators(&mut df)
        .map_err(|e| anyhow::anyhow!("ta-lib-in-rust failed: {e}"))?;

    let timestamps = extract_timestamps(&df)?
        .context("feature frame missing timestamp column")?;

    let mut names = Vec::new();
    let mut columns = Vec::new();

    if include_raw {
        for raw in ["open", "high", "low", "close", "volume"] {
            if let Ok(series) = df.column(raw) {
                names.push(raw.to_string());
                columns.push(series_to_f32(series, n_rows));
            }
        }
    }

    for series in df.get_columns() {
        let name = series.name();
        if name.eq_ignore_ascii_case("timestamp") {
            continue;
        }
        if original_cols.contains(name) {
            continue;
        }
        let mut col_name = String::from("ta_");
        col_name.push_str(name);
        names.push(col_name);
        columns.push(series_to_f32(series, n_rows));
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
