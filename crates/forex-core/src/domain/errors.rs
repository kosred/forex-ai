use thiserror::Error;

#[derive(Error, Debug)]
pub enum ForexBotError {
    #[error("Configuration error: {0}")]
    Configuration(String),

    #[error("Data error: {0}")]
    Data(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Trade execution error: {0}")]
    TradeExecution(String),

    #[error("API billing/quota error: {0}")]
    APIBilling(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, ForexBotError>;
