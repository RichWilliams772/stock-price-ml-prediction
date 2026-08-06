"""Data loading and chronological preparation.

Leakage controls
----------------
- Rows are always sorted by ``date`` ascending before any feature or target
  computation so later observations cannot silently precede earlier ones.
- Identifier columns (mic, symbol, isin, turnover, change percentages) are
  dropped early and never enter the feature matrix.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

# Columns retained from the raw Kaggle CSV for modeling.
PRICE_COLUMNS = [
    "date",
    "open_value",
    "high_value",
    "low_value",
    "last_value",
]

# Columns that identify the series or encode labels / future info — never used
# as model inputs.
IDENTIFIER_COLUMNS = [
    "mic",
    "symbol",
    "isin",
    "change_prev_close_percentage",
    "turnover",
]

DEFAULT_KAGGLE_DATASET = "satyajeetbedi/stock-price-dataset"
DEFAULT_LOCAL_CSV = Path(__file__).resolve().parents[1] / "data" / "stock_price_data.csv"


def download_dataset(dataset: str = DEFAULT_KAGGLE_DATASET) -> Path:
    """Download the Kaggle dataset via kagglehub and return its local path."""
    import kagglehub

    path = Path(kagglehub.dataset_download(dataset))
    return path


def find_csv_in_dir(directory: Path) -> Path:
    """Locate the first CSV file under ``directory`` (recursive)."""
    csv_files = sorted(directory.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {directory}")
    return csv_files[0]


def resolve_data_path(path: Optional[str | Path] = None) -> Path:
    """Resolve a usable CSV path: explicit path, local cache, or Kaggle download."""
    if path is not None:
        candidate = Path(path)
        if candidate.is_dir():
            return find_csv_in_dir(candidate)
        if not candidate.exists():
            raise FileNotFoundError(f"Data file not found: {candidate}")
        return candidate

    if DEFAULT_LOCAL_CSV.exists():
        return DEFAULT_LOCAL_CSV

    # Fall back to downloading from Kaggle when no local copy exists.
    downloaded = download_dataset()
    csv_path = find_csv_in_dir(downloaded)

    DEFAULT_LOCAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not DEFAULT_LOCAL_CSV.exists():
        import shutil

        shutil.copy2(csv_path, DEFAULT_LOCAL_CSV)
    return DEFAULT_LOCAL_CSV


def load_raw_data(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load the raw stock CSV without feature engineering."""
    csv_path = resolve_data_path(path)
    df = pd.read_csv(csv_path)
    return df


def prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Sort chronologically and keep only OHLC price columns.

    Leakage control: sorting by date ascending is required before rolling
    windows and target shifting so that past/future semantics are correct.
    """
    required = set(PRICE_COLUMNS)
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    # Chronological sort — never model on shuffled calendar order.
    out = out.sort_values("date", ascending=True).reset_index(drop=True)
    out = out[PRICE_COLUMNS].copy()
    # Drop incomplete OHLC rows before feature engineering.
    out = out.dropna(subset=["open_value", "high_value", "low_value", "last_value"])
    out = out.reset_index(drop=True)
    return out


def load_and_prepare(path: Optional[str | Path] = None) -> pd.DataFrame:
    """Load raw data and return a chronologically sorted price frame."""
    raw = load_raw_data(path)
    return prepare_price_frame(raw)


def dataset_metadata(df: pd.DataFrame, raw: Optional[pd.DataFrame] = None) -> dict:
    """Summarize dataset identity and date coverage for documentation."""
    meta = {
        "n_rows": int(len(df)),
        "date_min": str(pd.to_datetime(df["date"]).min().date()),
        "date_max": str(pd.to_datetime(df["date"]).max().date()),
        "columns": list(df.columns),
    }
    if raw is not None:
        for col in ("symbol", "mic", "isin"):
            if col in raw.columns:
                values = sorted(raw[col].dropna().unique().tolist())
                meta[col] = values[0] if len(values) == 1 else values
    return meta


def ensure_local_csv(path: Optional[str | Path] = None) -> Path:
    """Ensure a project-local CSV exists and return its path."""
    return resolve_data_path(path)


if __name__ == "__main__":
    csv = ensure_local_csv()
    raw = load_raw_data(csv)
    prepared = prepare_price_frame(raw)
    meta = dataset_metadata(prepared, raw)
    print(f"Cached CSV: {csv}")
    print(meta)
