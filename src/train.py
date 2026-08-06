"""Model training with chronological splits and leakage-safe preprocessing.

Leakage controls
----------------
- Train/test split is chronological (``shuffle=False``); no random mixing of
  future observations into the training set.
- Any scaling/preprocessing is fit on the training fold only.
- Feature columns exclude date, target, and identifiers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, TARGET_COLUMN, assert_no_target_leakage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42
TEST_SIZE = 0.2
RF_N_ESTIMATORS = 200


def chronological_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split a time-ordered frame into train/test without shuffling.

    Leakage control: the first ``(1 - test_size)`` fraction of rows (by time)
    forms the training set; the remainder is the hold-out test set.
    """
    if feature_columns is None:
        feature_columns = list(FEATURE_COLUMNS)
    assert_no_target_leakage(feature_columns)

    if "date" in df.columns and not df["date"].is_monotonic_increasing:
        raise ValueError("DataFrame must be sorted by date ascending before split.")

    n = len(df)
    if n < 10:
        raise ValueError(f"Not enough rows to split: {n}")

    split_idx = int(n * (1.0 - test_size))
    if split_idx <= 0 or split_idx >= n:
        raise ValueError(f"Invalid split index {split_idx} for n={n}")

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Ordering guarantee: every train date precedes every test date.
    if "date" in df.columns:
        if train_df["date"].max() > test_df["date"].min():
            raise ValueError("Train/test date ranges overlap — leakage risk.")

    X_train = train_df[feature_columns]
    X_test = test_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)
    y_test = test_df[TARGET_COLUMN].astype(int)
    train_dates = pd.to_datetime(train_df["date"])
    test_dates = pd.to_datetime(test_df["date"])
    return X_train, X_test, y_train, y_test, train_dates, test_dates


def build_random_forest(
    n_estimators: int = RF_N_ESTIMATORS,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    """Primary model — same algorithm as the original notebook."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )


def build_logistic_regression(random_state: int = RANDOM_STATE) -> Pipeline:
    """Simple linear baseline with train-only scaling (fit in Pipeline)."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def majority_class_baseline_predict(y_train: pd.Series, n_test: int) -> np.ndarray:
    """Predict the majority training class for every test row."""
    majority = int(y_train.value_counts().idxmax())
    return np.full(shape=n_test, fill_value=majority, dtype=int)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """Fit Random Forest (primary) plus logistic regression baseline."""
    assert_no_target_leakage(list(X_train.columns))

    rf = build_random_forest(random_state=random_state)
    rf.fit(X_train, y_train)

    logreg = build_logistic_regression(random_state=random_state)
    logreg.fit(X_train, y_train)

    return {
        "random_forest": rf,
        "logistic_regression": logreg,
        "majority_class": int(y_train.value_counts().idxmax()),
        "feature_columns": list(X_train.columns),
    }


def save_artifacts(
    artifacts: dict[str, Any],
    metadata: dict[str, Any],
    models_dir: Path | None = None,
) -> Path:
    """Persist trained models and metadata under ``models/``."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(artifacts["random_forest"], models_dir / "random_forest.joblib")
    joblib.dump(
        artifacts["logistic_regression"],
        models_dir / "logistic_regression.joblib",
    )
    payload = {
        **metadata,
        "majority_class": artifacts["majority_class"],
        "feature_columns": artifacts["feature_columns"],
    }
    with open(models_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    return models_dir


def load_artifacts(models_dir: Path | None = None) -> dict[str, Any]:
    """Load persisted models and metadata."""
    models_dir = Path(models_dir) if models_dir is not None else MODELS_DIR
    with open(models_dir / "metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return {
        "random_forest": joblib.load(models_dir / "random_forest.joblib"),
        "logistic_regression": joblib.load(models_dir / "logistic_regression.joblib"),
        "majority_class": metadata["majority_class"],
        "feature_columns": metadata["feature_columns"],
        "metadata": metadata,
    }


def run_training_pipeline(
    data_path: Optional[str | Path] = None,
    models_dir: Path | None = None,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, Any]:
    """End-to-end: load → features → chronological split → train → save."""
    from .data import dataset_metadata, load_and_prepare, load_raw_data
    from .features import build_feature_frame

    raw = load_raw_data(data_path)
    prices = load_and_prepare(data_path)
    featured = build_feature_frame(prices)

    X_train, X_test, y_train, y_test, train_dates, test_dates = (
        chronological_train_test_split(featured, test_size=test_size)
    )
    artifacts = train_models(X_train, y_train, random_state=random_state)

    meta = dataset_metadata(prices, raw)
    meta.update(
        {
            "n_featured_rows": int(len(featured)),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "train_date_min": str(train_dates.min().date()),
            "train_date_max": str(train_dates.max().date()),
            "test_date_min": str(test_dates.min().date()),
            "test_date_max": str(test_dates.max().date()),
            "test_size": test_size,
            "random_state": random_state,
            "rf_n_estimators": RF_N_ESTIMATORS,
            "target_definition": (
                "1 if next trading day's last_value > today's last_value, else 0"
            ),
            "ticker": meta.get("symbol", "CBX"),
            "exchange": meta.get("mic", "XZAG"),
        }
    )
    save_artifacts(artifacts, meta, models_dir=models_dir)

    return {
        "artifacts": artifacts,
        "metadata": meta,
        "featured": featured,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_dates": train_dates,
        "test_dates": test_dates,
    }


if __name__ == "__main__":
    result = run_training_pipeline()
    print("Training complete.")
    print(json.dumps(result["metadata"], indent=2, default=str))
