import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def have_cudf() -> bool:
    try:
        return True
    except Exception:
        return False


def have_polars() -> bool:
    try:
        return True
    except Exception:
        return False


def to_polars(df: pd.DataFrame):
    import polars as pl  # type: ignore

    return pl.from_pandas(df)


def to_cudf(df: pd.DataFrame):
    import cudf  # type: ignore

    return cudf.from_pandas(df)


def maybe_use_gpu_df(df: pd.DataFrame) -> Any:
    """
    Convert pandas -> cuDF if available; fallback to pandas.
    """
    if have_cudf():
        try:
            return to_cudf(df)
        except Exception as exc:
            logger.warning(f"cuDF conversion failed, using pandas: {exc}")
            return df
    return df


def maybe_use_polars(df: pd.DataFrame) -> Any:
    """
    Convert pandas -> Polars if available; fallback to pandas.
    """
    if have_polars():
        try:
            return to_polars(df)
        except Exception as exc:
            logger.warning(f"Polars conversion failed, using pandas: {exc}")
            return df
    return df
