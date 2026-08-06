"""Feature engineering and target construction.

Preserves the original notebook logic:
- Target: next-day close higher than today's close (binary direction)
- Features: open, high, low, last, MA10, MA50, daily return

Leakage controls
----------------
- Rolling means use only historical observations (pandas default
  ``rolling(window)`` is backward-looking; no center=True).
- Daily return uses ``pct_change()`` which references the prior close only.
- The target is built from ``shift(-1)`` and the final incomplete row is
  dropped via ``dropna`` so the model never trains on an unknown label.
- Identifier and target columns are excluded from ``FEATURE_COLUMNS``.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

FEATURE_COLUMNS = [
    "open_value",
    "high_value",
    "low_value",
    "last_value",
    "MA10",
    "MA50",
    "Return",
]

TARGET_COLUMN = "Target"
DATE_COLUMN = "date"

# Columns that must never appear in X.
EXCLUDED_FROM_FEATURES = {
    DATE_COLUMN,
    TARGET_COLUMN,
    "mic",
    "symbol",
    "isin",
    "change_prev_close_percentage",
    "turnover",
}


def add_target(df: pd.DataFrame, price_col: str = "last_value") -> pd.DataFrame:
    """Add next-day direction target.

    Target definition (same as original notebook):
        1 if tomorrow's close > today's close, else 0.

    Leakage control: the last row has no next-day close. We keep it as NaN
    (not coerced to 0) so ``dropna`` removes it. The original notebook used
    ``.astype(int)`` on a boolean comparison, which incorrectly labeled the
    final day as class 0 because ``NaN > x`` evaluates to False.
    """
    out = df.copy()
    if not out[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError(
            "Dates must be sorted ascending before creating the target "
            "(leakage control)."
        )

    next_close = out[price_col].shift(-1)
    # Preserve NaN on the final row so it is dropped with incomplete features.
    direction = next_close > out[price_col]
    out[TARGET_COLUMN] = direction.where(next_close.notna()).astype("float")
    return out


def add_technical_features(
    df: pd.DataFrame,
    price_col: str = "last_value",
    ma_windows: Iterable[int] = (10, 50),
) -> pd.DataFrame:
    """Add moving averages and daily return using only past observations.

    Leakage control: rolling windows are trailing (no future peeking). Rows
    whose windows are incomplete remain NaN until dropped by the caller.
    """
    out = df.copy()
    if not out[DATE_COLUMN].is_monotonic_increasing:
        raise ValueError(
            "Dates must be sorted ascending before feature generation "
            "(leakage control)."
        )

    for window in ma_windows:
        out[f"MA{window}"] = out[price_col].rolling(window=window).mean()

    # pct_change uses the previous close only — no look-ahead.
    out["Return"] = out[price_col].pct_change()
    return out


def drop_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with incomplete rolling windows or missing target."""
    return df.dropna().reset_index(drop=True)


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature + target pipeline from a prepared price frame."""
    out = add_target(df)
    out = add_technical_features(out)
    out = drop_incomplete_rows(out)
    out[TARGET_COLUMN] = out[TARGET_COLUMN].astype(int)
    return out


def get_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Return X with only approved feature columns (no target / identifiers)."""
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    leaked = [c for c in df.columns if c in EXCLUDED_FROM_FEATURES]
    # Safety: ensure we never silently include excluded columns.
    X = df[FEATURE_COLUMNS].copy()
    for col in X.columns:
        if col in EXCLUDED_FROM_FEATURES:
            raise ValueError(f"Leakage risk: excluded column in features: {col}")
    _ = leaked  # documented for reviewers; X already filtered
    return X


def get_target(df: pd.DataFrame) -> pd.Series:
    """Return the binary direction target."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}'")
    return df[TARGET_COLUMN].astype(int)


def assert_no_target_leakage(feature_columns: list[str]) -> None:
    """Raise if the target or date/id columns appear in the feature list."""
    bad = set(feature_columns) & EXCLUDED_FROM_FEATURES
    if bad:
        raise AssertionError(f"Target/identifier leakage in features: {sorted(bad)}")
