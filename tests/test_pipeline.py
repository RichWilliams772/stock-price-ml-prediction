"""Focused tests for chronological integrity, features, and leakage controls."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import prepare_price_frame
from src.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    add_target,
    add_technical_features,
    assert_no_target_leakage,
    build_feature_frame,
    drop_incomplete_rows,
    get_feature_matrix,
)
from src.predict import validate_feature_frame
from src.train import chronological_train_test_split


def _sample_prices(n: int = 80, start: str = "2010-01-04") -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.bdate_range(start=start, periods=n)
    # Random-walk-like closes so rolling features are well-defined.
    closes = 2000 + np.cumsum(rng.normal(0, 5, size=n))
    opens = closes + rng.normal(0, 1, size=n)
    highs = np.maximum(opens, closes) + rng.uniform(0, 3, size=n)
    lows = np.minimum(opens, closes) - rng.uniform(0, 3, size=n)
    return pd.DataFrame(
        {
            "date": dates,
            "open_value": opens,
            "high_value": highs,
            "low_value": lows,
            "last_value": closes,
        }
    )


def test_prepare_price_frame_sorts_chronologically():
    df = _sample_prices(30)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    # Add decoy identifier columns that must be dropped.
    shuffled["symbol"] = "CBX"
    prepared = prepare_price_frame(shuffled)
    assert prepared["date"].is_monotonic_increasing
    assert list(prepared.columns) == [
        "date",
        "open_value",
        "high_value",
        "low_value",
        "last_value",
    ]


def test_target_shifting_uses_next_day_close():
    df = _sample_prices(10)
    out = add_target(df)
    # For each complete row i, target is 1 iff close[i+1] > close[i].
    for i in range(len(df) - 1):
        expected = int(df.loc[i + 1, "last_value"] > df.loc[i, "last_value"])
        assert int(out.loc[i, TARGET_COLUMN]) == expected
    # Final row has no next-day close → NaN (not silently 0).
    assert pd.isna(out.loc[len(df) - 1, TARGET_COLUMN])


def test_feature_generation_rolling_uses_history_only():
    df = _sample_prices(60)
    featured = add_technical_features(df)
    # First 9 MA10 values incomplete; index 9 equals mean of first 10 closes.
    assert pd.isna(featured.loc[8, "MA10"])
    expected_ma10 = df["last_value"].iloc[:10].mean()
    assert featured.loc[9, "MA10"] == pytest.approx(expected_ma10)
    # Return at t uses only close[t] and close[t-1].
    assert featured.loc[1, "Return"] == pytest.approx(
        df["last_value"].iloc[1] / df["last_value"].iloc[0] - 1
    )


def test_absence_of_target_leakage_in_feature_matrix():
    featured = build_feature_frame(_sample_prices(80))
    X = get_feature_matrix(featured)
    assert TARGET_COLUMN not in X.columns
    assert "date" not in X.columns
    assert list(X.columns) == FEATURE_COLUMNS
    assert_no_target_leakage(list(X.columns))
    with pytest.raises(AssertionError):
        assert_no_target_leakage(["MA10", TARGET_COLUMN])


def test_train_test_ordering_is_chronological():
    featured = build_feature_frame(_sample_prices(100))
    X_train, X_test, y_train, y_test, train_dates, test_dates = (
        chronological_train_test_split(featured, test_size=0.2)
    )
    assert len(X_train) + len(X_test) == len(featured)
    assert train_dates.max() < test_dates.min()
    assert train_dates.is_monotonic_increasing
    assert test_dates.is_monotonic_increasing


def test_expected_model_input_schema():
    featured = build_feature_frame(_sample_prices(80))
    X = get_feature_matrix(featured)
    validated = validate_feature_frame(X)
    assert list(validated.columns) == FEATURE_COLUMNS
    with pytest.raises(ValueError):
        validate_feature_frame(X.drop(columns=["Return"]))


def test_drop_incomplete_rows_removes_warmup_and_final_target():
    df = add_technical_features(add_target(_sample_prices(60)))
    cleaned = drop_incomplete_rows(df)
    assert cleaned[TARGET_COLUMN].notna().all()
    assert cleaned["MA50"].notna().all()
    assert cleaned["Return"].notna().all()
    # Warm-up for MA50 is 49 rows; plus final missing target → at least 50 dropped.
    assert len(cleaned) <= len(df) - 50
