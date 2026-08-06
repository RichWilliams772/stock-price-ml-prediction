"""Evaluation metrics, plots, and baseline comparisons.

Reports only metrics computed from model outputs. Does not claim forecasting
skill beyond what the numbers show.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .features import FEATURE_COLUMNS
from .train import majority_class_baseline_predict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGES_DIR = PROJECT_ROOT / "docs" / "images"
MODELS_DIR = PROJECT_ROOT / "models"


def compute_classification_metrics(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    """Compute accuracy, precision, recall, F1, and optional ROC-AUC."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "n_samples": int(len(y_true)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, zero_division=0, output_dict=True
        ),
    }

    # ROC-AUC only when probabilities exist and both classes are present.
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["roc_auc"] = None

    return metrics


def evaluate_models(
    artifacts: dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_train: pd.Series,
) -> dict[str, Any]:
    """Evaluate Random Forest, logistic regression, and majority baseline."""
    results: dict[str, Any] = {}

    rf = artifacts["random_forest"]
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = compute_classification_metrics(y_test, rf_pred, rf_proba)
    results["random_forest"]["predictions"] = rf_pred.tolist()
    results["random_forest"]["probabilities"] = rf_proba.tolist()

    lr = artifacts["logistic_regression"]
    lr_pred = lr.predict(X_test)
    lr_proba = lr.predict_proba(X_test)[:, 1]
    results["logistic_regression"] = compute_classification_metrics(
        y_test, lr_pred, lr_proba
    )
    results["logistic_regression"]["predictions"] = lr_pred.tolist()

    maj_pred = majority_class_baseline_predict(y_train, len(y_test))
    results["majority_class"] = compute_classification_metrics(y_test, maj_pred)
    results["majority_class"]["predictions"] = maj_pred.tolist()
    results["majority_class"]["majority_label"] = int(artifacts["majority_class"])

    # Feature importance from the primary Random Forest.
    importances = getattr(rf, "feature_importances_", None)
    if importances is not None:
        fi = (
            pd.DataFrame(
                {"Feature": list(X_test.columns), "Importance": importances}
            )
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        results["feature_importance"] = fi.to_dict(orient="records")

    return results


def rolling_directional_accuracy(
    y_true,
    y_pred,
    window: int = 30,
) -> pd.Series:
    """Rolling accuracy of direction predictions over the test period."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    return pd.Series(correct).rolling(window=window, min_periods=max(5, window // 5)).mean()


def _save_fig(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_price_history(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    dates = pd.to_datetime(df["date"])
    ax.plot(dates, df["last_value"], color="#1f4e79", linewidth=1.2)
    ax.set_title("CBX (Zagreb) Closing Price History")
    ax.set_xlabel("Date")
    ax.set_ylabel("Last value")
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return _save_fig(fig, out_dir / "stock-price-history.png")


def plot_technical_indicators(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    dates = pd.to_datetime(df["date"])

    axes[0].plot(dates, df["last_value"], label="Close", color="#1f4e79", linewidth=1.1)
    if "MA10" in df.columns:
        axes[0].plot(dates, df["MA10"], label="MA10", color="#c45c26", linewidth=1.0)
    if "MA50" in df.columns:
        axes[0].plot(dates, df["MA50"], label="MA50", color="#2a9d8f", linewidth=1.0)
    axes[0].set_title("Price and Moving Averages")
    axes[0].set_ylabel("Price")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    if "Return" in df.columns:
        axes[1].plot(dates, df["Return"], color="#4a4a4a", linewidth=0.8)
        axes[1].axhline(0, color="black", linewidth=0.6)
        axes[1].set_title("Daily Return")
        axes[1].set_ylabel("Return")
        axes[1].grid(True, alpha=0.3)

    axes[1].set_xlabel("Date")
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_fig(fig, out_dir / "technical-indicators.png")


def plot_class_distribution(y: pd.Series, out_dir: Path) -> Path:
    counts = y.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Down (0)", "Up (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    bars = ax.bar(labels, values, color=["#8b0000", "#2e7d32"])
    ax.set_title("Target Class Distribution (Next-Day Direction)")
    ax.set_ylabel("Count")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
        )
    ax.grid(True, axis="y", alpha=0.3)
    return _save_fig(fig, out_dir / "class-distribution.png")


def plot_confusion_matrix(cm: list[list[int]], out_dir: Path) -> Path:
    matrix = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Down", "Pred Up"])
    ax.set_yticklabels(["Actual Down", "Actual Up"])
    ax.set_title("Random Forest Confusion Matrix (Test Set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    return _save_fig(fig, out_dir / "confusion-matrix.png")


def plot_feature_importance(records: list[dict], out_dir: Path) -> Path:
    fi = pd.DataFrame(records).sort_values("Importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(fi["Feature"], fi["Importance"], color="#1f4e79")
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(True, axis="x", alpha=0.3)
    return _save_fig(fig, out_dir / "feature-importance.png")


def plot_model_comparison(results: dict[str, Any], out_dir: Path) -> Path:
    names = ["majority_class", "logistic_regression", "random_forest"]
    labels = ["Majority Class", "Logistic Regression", "Random Forest"]
    metrics_keys = ["accuracy", "precision", "recall", "f1"]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, key in enumerate(metrics_keys):
        vals = [results[n][key] for n in names]
        ax.bar(x + (i - 1.5) * width, vals, width, label=key)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison on Chronological Test Set")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    return _save_fig(fig, out_dir / "model-comparison.png")


def plot_predictions_over_time(
    dates,
    y_true,
    y_pred,
    out_dir: Path,
    window: int = 30,
) -> Path:
    dates = pd.to_datetime(dates)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    correct = (y_true == y_pred).astype(float)
    rolling = (
        pd.Series(correct, index=dates)
        .rolling(window=window, min_periods=max(5, window // 5))
        .mean()
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].step(dates, y_true, where="mid", label="Actual", alpha=0.8)
    axes[0].step(dates, y_pred, where="mid", label="Predicted", alpha=0.8)
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["Down", "Up"])
    axes[0].set_title("Predicted vs Actual Next-Day Direction (Test Period)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(rolling.index, rolling.values, color="#1f4e79", linewidth=1.3)
    axes[1].axhline(0.5, color="gray", linestyle="--", linewidth=0.9, label="0.50 chance")
    axes[1].set_title(f"Rolling Directional Accuracy ({window}-day window)")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_xlabel("Date")
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    return _save_fig(fig, out_dir / "predictions-over-time.png")


def generate_all_figures(
    featured: pd.DataFrame,
    results: dict[str, Any],
    y_test: pd.Series,
    test_dates: pd.Series,
    out_dir: Path | None = None,
) -> list[Path]:
    """Create polished evaluation figures under docs/images/."""
    out_dir = Path(out_dir) if out_dir is not None else IMAGES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_price_history(featured, out_dir),
        plot_technical_indicators(featured, out_dir),
        plot_class_distribution(featured["Target"], out_dir),
        plot_confusion_matrix(results["random_forest"]["confusion_matrix"], out_dir),
        plot_feature_importance(results["feature_importance"], out_dir),
        plot_model_comparison(results, out_dir),
        plot_predictions_over_time(
            test_dates,
            y_test,
            results["random_forest"]["predictions"],
            out_dir,
        ),
    ]
    return paths


def save_metrics(results: dict[str, Any], path: Path | None = None) -> Path:
    """Persist summary metrics (without bulky prediction arrays) to JSON."""
    path = path or (MODELS_DIR / "metrics.json")
    path.parent.mkdir(parents=True, exist_ok=True)

    summary = {}
    for name in ("majority_class", "logistic_regression", "random_forest"):
        block = results[name]
        summary[name] = {
            k: block[k]
            for k in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "n_samples",
                "confusion_matrix",
                "classification_report",
            )
            if k in block
        }
        if name == "majority_class" and "majority_label" in block:
            summary[name]["majority_label"] = block["majority_label"]

    summary["feature_importance"] = results.get("feature_importance", [])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return path


def run_evaluation(
    train_result: dict[str, Any] | None = None,
    images_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full evaluation from a training result dict or by retraining."""
    if train_result is None:
        from .train import run_training_pipeline

        train_result = run_training_pipeline()

    results = evaluate_models(
        train_result["artifacts"],
        train_result["X_test"],
        train_result["y_test"],
        train_result["y_train"],
    )
    paths = generate_all_figures(
        train_result["featured"],
        results,
        train_result["y_test"],
        train_result["test_dates"],
        out_dir=images_dir,
    )
    metrics_path = save_metrics(results)

    # Also store test predictions for the Streamlit app / notebook.
    pred_frame = pd.DataFrame(
        {
            "date": pd.to_datetime(train_result["test_dates"]).astype(str),
            "actual": train_result["y_test"].to_numpy(),
            "predicted": results["random_forest"]["predictions"],
            "probability_up": results["random_forest"]["probabilities"],
        }
    )
    pred_path = MODELS_DIR / "test_predictions.csv"
    pred_frame.to_csv(pred_path, index=False)

    return {
        "results": results,
        "figure_paths": paths,
        "metrics_path": metrics_path,
        "predictions_path": pred_path,
        "metadata": train_result["metadata"],
    }


if __name__ == "__main__":
    from .train import run_training_pipeline

    train_result = run_training_pipeline()
    evaluation = run_evaluation(train_result)
    rf = evaluation["results"]["random_forest"]
    print("Random Forest accuracy:", rf["accuracy"])
    print("Figures:", [str(p) for p in evaluation["figure_paths"]])
