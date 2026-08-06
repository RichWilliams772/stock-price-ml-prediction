"""CLI entry: train models, evaluate, and write figures/metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import run_evaluation
from src.train import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train and evaluate next-day stock direction classifiers. "
            "Educational/research use only — not financial advice."
        )
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Optional path to CSV (default: data/stock_price_data.csv or Kaggle download).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Chronological hold-out fraction (default: 0.2).",
    )
    args = parser.parse_args()

    train_result = run_training_pipeline(
        data_path=args.data,
        test_size=args.test_size,
    )
    evaluation = run_evaluation(train_result)

    summary = {
        "metadata": evaluation["metadata"],
        "random_forest": {
            k: evaluation["results"]["random_forest"][k]
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "n_samples")
        },
        "logistic_regression": {
            k: evaluation["results"]["logistic_regression"][k]
            for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "n_samples")
        },
        "majority_class": {
            k: evaluation["results"]["majority_class"][k]
            for k in ("accuracy", "precision", "recall", "f1", "n_samples")
        },
        "figures": [str(p) for p in evaluation["figure_paths"]],
    }
    print(json.dumps(summary, indent=2, default=str))
    print(
        "\nDisclaimer: This project is for educational and research purposes only. "
        "It is not financial advice and should not be used to make investment decisions."
    )


if __name__ == "__main__":
    main()
