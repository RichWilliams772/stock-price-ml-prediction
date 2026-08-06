"""Inference helpers for the trained direction classifier.

This module scores historical feature rows. It is not a live trading system
and does not provide investment advice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS, assert_no_target_leakage
from .train import load_artifacts

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


def validate_feature_frame(X: pd.DataFrame, expected_columns: list[str] | None = None) -> pd.DataFrame:
    """Ensure the input matches the trained model schema."""
    expected = expected_columns or list(FEATURE_COLUMNS)
    assert_no_target_leakage(list(X.columns))
    missing = [c for c in expected if c not in X.columns]
    extra_blocked = [c for c in X.columns if c not in expected]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")
    # Allow column subsetting to the expected schema; ignore unexpected cols.
    return X[expected].copy()


def predict_direction(
    X: pd.DataFrame,
    models_dir: Path | None = None,
    model_name: str = "random_forest",
) -> dict[str, Any]:
    """Predict next-day direction for rows in ``X``.

    Returns class labels and, when available, P(up).
    """
    artifacts = load_artifacts(models_dir)
    expected = artifacts["feature_columns"]
    X_valid = validate_feature_frame(X, expected)

    model = artifacts[model_name]
    labels = model.predict(X_valid)
    out: dict[str, Any] = {
        "predictions": np.asarray(labels),
        "feature_columns": expected,
        "model_name": model_name,
    }
    if hasattr(model, "predict_proba"):
        out["probabilities"] = model.predict_proba(X_valid)[:, 1]
    return out


def predict_from_featured_frame(
    featured: pd.DataFrame,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Score a featured DataFrame and return predictions aligned with dates."""
    artifacts = load_artifacts(models_dir)
    X = validate_feature_frame(featured, artifacts["feature_columns"])
    result = predict_direction(X, models_dir=models_dir)
    frame = pd.DataFrame(
        {
            "date": featured["date"].values if "date" in featured.columns else range(len(X)),
            "predicted_direction": result["predictions"],
        }
    )
    if "probabilities" in result:
        frame["probability_up"] = result["probabilities"]
    if "Target" in featured.columns:
        frame["actual_direction"] = featured["Target"].values
    return frame


if __name__ == "__main__":
    from .data import load_and_prepare
    from .features import build_feature_frame

    featured = build_feature_frame(load_and_prepare())
    # Score the last 5 complete rows as a smoke check.
    preds = predict_from_featured_frame(featured.tail(5))
    print(preds.to_string(index=False))
    print(
        "\nDisclaimer: Educational/research only. Not financial advice."
    )
