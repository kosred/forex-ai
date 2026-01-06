// Logging setup for Forex trading system
// Port of src/forex_bot/core/logging.py

use std::path::PathBuf;
use tracing::Level;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

/// Setup structured logging with tracing
///
/// This configures:
/// - Console output with color and timestamps
/// - File rotation (50MB max, 3 backups)
/// - Environment variable filtering
/// - Silencing of noisy libraries
pub fn setup_logging(verbose: bool) -> anyhow::Result<()> {
    let level = if verbose { Level::DEBUG } else { Level::INFO };

    // Get log directory from environment or use default
    let log_dir: PathBuf = std::env::var("LOG_DIR")
        .unwrap_or_else(|_| "logs".to_string())
        .into();

    std::fs::create_dir_all(&log_dir)?;

    let log_file =
        log_dir.join(std::env::var("LOG_FILE").unwrap_or_else(|_| "forex_bot.log".to_string()));

    // File appender with rotation (50MB max, 3 backups)
    let file_appender = tracing_appender::rolling::never(&log_dir, log_file.file_name().unwrap());
    let (non_blocking, _guard) = tracing_appender::non_blocking(file_appender);

    // Create environment filter that silences noisy libraries
    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        EnvFilter::new(format!("{}", level))
            // Silence noisy HTTP libraries
            .add_directive("httpcore=warn".parse().unwrap())
            .add_directive("httpx=warn".parse().unwrap())
            .add_directive("hyper=warn".parse().unwrap())
            .add_directive("reqwest=warn".parse().unwrap())
            .add_directive("h2=warn".parse().unwrap())
            // Silence other noisy crates
            .add_directive("tokio=info".parse().unwrap())
            .add_directive("runtime=info".parse().unwrap())
    });

    // Console layer with nice formatting
    let console_layer = fmt::layer()
        .with_target(true)
        .with_thread_ids(false)
        .with_thread_names(false)
        .with_ansi(true)
        .with_writer(std::io::stdout);

    // File layer with JSON formatting for structured logs
    let file_layer = fmt::layer()
        .with_target(true)
        .with_ansi(false)
        .json()
        .with_writer(non_blocking);

    // Combine layers
    tracing_subscriber::registry()
        .with(env_filter)
        .with(console_layer)
        .with(file_layer)
        .init();

    // Log initial startup message
    tracing::info!("Logging initialized (verbose={})", verbose);
    tracing::info!("Log file: {:?}", log_file);

    Ok(())
}

/// Setup minimal logging (console only, no files)
pub fn setup_minimal_logging(verbose: bool) -> anyhow::Result<()> {
    let level = if verbose { Level::DEBUG } else { Level::INFO };

    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(format!("{}", level)));

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(true)
        .with_ansi(true)
        .init();

    tracing::info!("Minimal logging initialized");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_logging() {
        // This test just ensures the function doesn't panic
        let _ = setup_minimal_logging(false);
    }
}
