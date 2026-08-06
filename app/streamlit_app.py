"""Polished research dashboard for next-day direction classification.

Presentation layer only — does not retrain models or alter evaluation methodology.
Educational / research display; not a trading tool and not financial advice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, APP_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from app.components import (  # type: ignore
        DISCLAIMER,
        chart_class_distribution,
        chart_confusion_matrix,
        chart_daily_returns,
        chart_feature_importance,
        chart_model_comparison,
        chart_moving_averages,
        chart_prediction_analysis,
        chart_price_history,
        comparison_table_for_display,
        fmt_int,
        fmt_pct,
        inject_styles,
        kpi_card_html,
        kpi_row,
        positive_class_share,
        readable_feature,
        section_header,
    )
except ImportError:
    from components import (
        DISCLAIMER,
        chart_class_distribution,
        chart_confusion_matrix,
        chart_daily_returns,
        chart_feature_importance,
        chart_model_comparison,
        chart_moving_averages,
        chart_prediction_analysis,
        chart_price_history,
        comparison_table_for_display,
        fmt_int,
        fmt_pct,
        inject_styles,
        kpi_card_html,
        kpi_row,
        positive_class_share,
        readable_feature,
        section_header,
    )

from src.data import load_and_prepare
from src.features import FEATURE_COLUMNS, build_feature_frame

IMAGES = PROJECT_ROOT / "docs" / "images"
MODELS = PROJECT_ROOT / "models"


@st.cache_data(show_spinner=False)
def load_json(path_str: str) -> dict | None:
    path = Path(path_str)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_featured_frame() -> pd.DataFrame:
    return build_feature_frame(load_and_prepare())


@st.cache_data(show_spinner=False)
def load_predictions() -> pd.DataFrame | None:
    path = MODELS / "test_predictions.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero-wrap">
          <div class="hero-kicker">Financial machine learning · Research dashboard</div>
          <h1 class="hero-title">Stock Price Movement — Direction Classification</h1>
          <p class="hero-subtitle">
            A time-series machine learning study of next-day stock direction using
            historical OHLC data and technical features.
          </p>
          <div class="disclaimer-banner"><strong>Disclaimer:</strong> {DISCLAIMER}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dataset_kpis(metadata: dict, featured: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "Dataset snapshot",
            "Values come from the prepared CBX / XZAG series and feature pipeline.",
            label="Overview",
        ),
        unsafe_allow_html=True,
    )
    pos_share = positive_class_share(featured)
    cards = [
        kpi_card_html(
            "Ticker / Market",
            f"{metadata.get('ticker', 'CBX')} / {metadata.get('exchange', 'XZAG')}",
            "Zagreb Stock Exchange index series",
            accent=True,
        ),
        kpi_card_html(
            "Date range",
            f"{metadata.get('date_min')} → {metadata.get('date_max')}",
            "Full price history before warm-up drop",
        ),
        kpi_card_html("Featured rows", fmt_int(metadata.get("n_featured_rows")), "After rolling warm-up & target drop"),
        kpi_card_html("Training rows", fmt_int(metadata.get("n_train")), "Earlier chronological segment"),
        kpi_card_html("Test rows", fmt_int(metadata.get("n_test")), "Later chronological holdout"),
        kpi_card_html(
            "Positive-class share",
            fmt_pct(pos_share),
            "Share of Up days in featured data",
        ),
    ]
    st.markdown(kpi_row(cards), unsafe_allow_html=True)
    st.markdown(
        """
        <div class="panel">
          <h4>Target definition</h4>
          <p>
            This project classifies <strong>next-day direction</strong>
            (<code>1</code> if tomorrow’s close &gt; today’s close, else <code>0</code>).
            It is not price-level forecasting.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_kpis(metrics: dict) -> None:
    rf = metrics["random_forest"]
    st.markdown(
        section_header(
            "Primary model metrics",
            "All scores below are from the chronological holdout set (not a random split).",
            label="Holdout evaluation",
        ),
        unsafe_allow_html=True,
    )
    # Near-chance performance → caution styling, not green "success".
    cards = [
        kpi_card_html("Primary model", "Random Forest", "n_estimators = 200 · research primary", accent=True),
        kpi_card_html("Accuracy", fmt_pct(rf["accuracy"]), "Below majority baseline", caution=True),
        kpi_card_html("Precision", fmt_pct(rf["precision"]), "Positive class (Up)", caution=True),
        kpi_card_html("Recall", fmt_pct(rf["recall"]), "Positive class (Up)", caution=True),
        kpi_card_html("F1 score", fmt_pct(rf["f1"]), "Positive class (Up)", caution=True),
        kpi_card_html("ROC-AUC", fmt_pct(rf["roc_auc"]), "Close to chance (≈50%)", caution=True),
    ]
    st.markdown(kpi_row(cards), unsafe_allow_html=True)

    maj_acc = metrics["majority_class"]["accuracy"]
    rf_acc = rf["accuracy"]
    rf_recall = rf["recall"]
    rf_auc = rf["roc_auc"]
    st.markdown(
        f"""
        <div class="panel caution">
          <h4>How to read these results</h4>
          <ul>
            <li>The majority-class baseline reached <strong>{fmt_pct(maj_acc)}</strong> accuracy,
                higher than Random Forest (<strong>{fmt_pct(rf_acc)}</strong>).</li>
            <li>Random Forest positive-class recall was <strong>{fmt_pct(rf_recall)}</strong>,
                which is higher than its overall accuracy alone would suggest — but precision remained weak.</li>
            <li>ROC-AUC was <strong>{fmt_pct(rf_auc)}</strong>, close to 0.5 (chance-level ranking).</li>
            <li>These results demonstrate an end-to-end ML workflow on financial time series;
                they do <strong>not</strong> show dependable market predictability.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_overview(metadata: dict, metrics: dict, featured: pd.DataFrame) -> None:
    render_dataset_kpis(metadata, featured)
    render_model_kpis(metrics)

    st.markdown(
        section_header(
            "Workflow at a glance",
            "Chronological preparation → technical features → holdout evaluation.",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="panel">
          <ol>
            <li>Load CBX / XZAG daily OHLC prices and sort chronologically.</li>
            <li>Engineer trailing MA10 / MA50 and daily return (history-only windows).</li>
            <li>Label next-day direction; drop incomplete warm-up / unlabeled rows.</li>
            <li>Split chronologically
                ({metadata.get('train_date_min')}–{metadata.get('train_date_max')} train ·
                 {metadata.get('test_date_min')}–{metadata.get('test_date_max')} test).</li>
            <li>Compare Random Forest against majority-class and logistic-regression baselines.</li>
          </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_market_data(featured: pd.DataFrame) -> None:
    st.markdown(
        section_header(
            "Market data",
            "Interactive views of the same series used for modeling. Static PNG exports remain under docs/images/.",
            label="Exploration",
        ),
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_price_history(featured), width='stretch')
    st.caption("Saved figure: `docs/images/stock-price-history.png`")

    st.plotly_chart(chart_moving_averages(featured), width='stretch')
    st.caption("Trailing averages only — no centered windows. Saved: `docs/images/technical-indicators.png`")

    st.plotly_chart(chart_daily_returns(featured), width='stretch')
    st.caption("Daily percent change of closing price.")

    st.plotly_chart(chart_class_distribution(featured["Target"]), width='stretch')
    st.caption("Saved figure: `docs/images/class-distribution.png`")


def render_model_evaluation(metrics: dict) -> None:
    st.markdown(
        section_header(
            "Model evaluation",
            "Chronological holdout comparison. Random Forest is the primary research model, not automatically the best scorer.",
            label="Evaluation",
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="panel">
          <h4>Why baseline comparison matters</h4>
          <p>
            A classifier can look reasonable in isolation while failing to beat a trivial rule
            (here: always predict the majority training class). Baseline comparison keeps the
            research claim honest when signal is weak.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    table = comparison_table_for_display(metrics)
    st.markdown("##### Comparison table")
    st.caption(
        "Percentages from saved chronological-holdout metrics. "
        "★ marks the strongest value in each column. "
        "Random Forest is the primary research model, not necessarily the top scorer."
    )
    st.dataframe(table, width='stretch', hide_index=True)

    st.plotly_chart(chart_model_comparison(metrics), width='stretch')
    st.caption("Saved figure: `docs/images/model-comparison.png`")

    st.markdown("##### Confusion matrix")
    cm = metrics["random_forest"]["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    st.markdown(
        f"""
        <div class="cm-grid">
          <div class="cm-cell good">
            <div class="cm-name">True Down</div>
            <div class="cm-value">{tn}</div>
            <div class="cm-desc">Actual Down, predicted Down — correct negative.</div>
          </div>
          <div class="cm-cell bad">
            <div class="cm-name">False Up</div>
            <div class="cm-value">{fp}</div>
            <div class="cm-desc">Actual Down, predicted Up — false positive.</div>
          </div>
          <div class="cm-cell bad">
            <div class="cm-name">False Down</div>
            <div class="cm-value">{fn}</div>
            <div class="cm-desc">Actual Up, predicted Down — false negative.</div>
          </div>
          <div class="cm-cell good">
            <div class="cm-name">True Up</div>
            <div class="cm-value">{tp}</div>
            <div class="cm-desc">Actual Up, predicted Up — correct positive.</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_confusion_matrix(cm), width='stretch')
    st.caption("Underlying counts unchanged from the saved evaluation. Figure: `docs/images/confusion-matrix.png`")

    st.markdown("##### Top Predictive Features")
    st.markdown(
        """
        <div class="panel">
          <p>
            Feature importance reflects how the fitted Random Forest used the variables.
            It does not establish causal market influence.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.plotly_chart(chart_feature_importance(metrics["feature_importance"]), width='stretch')
    st.caption("Importance values unchanged. Saved figure: `docs/images/feature-importance.png`")

    with st.expander("Readable feature name mapping"):
        mapping = pd.DataFrame(
            [{"Raw name": r["Feature"], "Display label": readable_feature(r["Feature"]), "Importance": r["Importance"]}
             for r in metrics["feature_importance"]]
        )
        st.dataframe(mapping, width='stretch', hide_index=True)


def render_prediction_analysis(featured: pd.DataFrame, preds: pd.DataFrame | None) -> None:
    st.markdown(
        section_header(
            "Prediction analysis",
            "Test-period behavior of the saved Random Forest scores. Historical research context only.",
            label="Test period",
        ),
        unsafe_allow_html=True,
    )

    if preds is None:
        st.warning("Test predictions not found. Run `python run_pipeline.py` to generate them.")
        return

    st.plotly_chart(chart_prediction_analysis(preds), width='stretch')
    st.caption(
        "Markers show actual direction and correct/incorrect outcomes against P(Up). "
        "Dashed line is the 0.50 decision threshold. Saved companion figure: `docs/images/predictions-over-time.png`"
    )

    with st.expander("Raw test-period prediction table (last 40 rows)"):
        show = preds.copy()
        show["actual"] = show["actual"].map({0: "Down", 1: "Up"})
        show["predicted"] = show["predicted"].map({0: "Down", 1: "Up"})
        show["correct"] = show["actual"] == show["predicted"]
        st.dataframe(show.tail(40), width='stretch', hide_index=True)

    st.markdown("##### Research Prediction Explorer")
    st.markdown(
        """
        <div class="panel caution">
          <h4>Historical inspection only</h4>
          <p>
            Select an existing row from the chronological test set to inspect model inputs and
            saved outputs. This is <strong>not</strong> a live trading tool, does not accept
            today’s market prices, and does not issue Buy / Sell / Hold recommendations.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Join featured features onto test predictions by date for the explorer.
    feat = featured.copy()
    feat["date"] = pd.to_datetime(feat["date"])
    merged = preds.merge(feat, on="date", how="left", suffixes=("", "_feat"))

    date_options = merged["date"].dt.strftime("%Y-%m-%d").tolist()
    selected = st.selectbox(
        "Select a historical test date",
        options=date_options,
        index=len(date_options) - 1,
        help="Dates are from the chronological holdout set only.",
    )
    row = merged.loc[merged["date"] == pd.Timestamp(selected)].iloc[0]

    actual = "Up" if int(row["actual"]) == 1 else "Down"
    predicted = "Up" if int(row["predicted"]) == 1 else "Down"
    correct = int(row["actual"]) == int(row["predicted"])
    prob = float(row["probability_up"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Date", selected)
    c2.metric("Predicted direction", predicted)
    c3.metric("Probability of Up", fmt_pct(prob))
    c4.metric("Actual direction", actual, delta="Correct" if correct else "Incorrect", delta_color="off")

    feature_rows = []
    for col in FEATURE_COLUMNS:
        if col in row.index and pd.notna(row[col]):
            feature_rows.append(
                {"Feature": readable_feature(col), "Raw name": col, "Value": float(row[col])}
            )
    st.markdown("**Input features for the selected day**")
    st.dataframe(pd.DataFrame(feature_rows), width='stretch', hide_index=True)
    st.caption(
        "Predicted direction uses the saved 0.50 probability threshold on historical test rows. "
        "Do not treat this as a forecast for a future trading day outside this research sample."
    )


def render_limitations() -> None:
    st.markdown(
        section_header(
            "Limitations & research takeaways",
            "Honest constraints and what the saved results actually support.",
            label="Integrity",
        ),
        unsafe_allow_html=True,
    )

    st.markdown("##### Key Research Takeaways")
    st.markdown(
        """
        <div class="panel">
          <ul class="takeaway-list">
            <li>Next-day direction was difficult to classify with this OHLC + moving-average feature set.</li>
            <li>The majority-class baseline remained competitive and beat Random Forest on accuracy.</li>
            <li>Random Forest captured some positive-class recall but did not improve overall accuracy.</li>
            <li>Feature importance was spread fairly evenly across price levels and technical indicators.</li>
            <li>Performance varied over time, as shown by rolling directional accuracy on the test period.</li>
            <li>Stronger validation design and richer features would be required before any practical use.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            """
            <div class="panel">
              <h4>Modeling limitations</h4>
              <ul>
                <li>Simple technical features may not encode useful directional signal.</li>
                <li>Tree ensembles can overfit noisy financial labels.</li>
                <li>Feature importance is not a causal explanation.</li>
              </ul>
            </div>
            <div class="panel">
              <h4>Market limitations</h4>
              <ul>
                <li>Daily returns are noisy and weakly predictable.</li>
                <li>Markets are non-stationary; regimes change.</li>
                <li>Past performance does not guarantee future performance.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            """
            <div class="panel">
              <h4>Evaluation limitations</h4>
              <ul>
                <li>Single chronological holdout; no walk-forward nested tuning shown here.</li>
                <li>Mild class imbalance can make naive baselines look competitive.</li>
                <li>Leakage is mitigated here but must be re-checked when features change.</li>
              </ul>
            </div>
            <div class="panel">
              <h4>Practical limitations</h4>
              <ul>
                <li>Transaction costs, slippage, and liquidity are not modeled.</li>
                <li>No brokerage integration or live market API.</li>
                <li>Outputs are research diagnostics, not trading signals.</li>
              </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="disclaimer-banner" style="margin-top:0.6rem;">
          <strong>Disclaimer:</strong> {DISCLAIMER}
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Static PNG gallery (docs/images)"):
        for name in [
            "stock-price-history.png",
            "technical-indicators.png",
            "class-distribution.png",
            "model-comparison.png",
            "confusion-matrix.png",
            "feature-importance.png",
            "predictions-over-time.png",
        ]:
            path = IMAGES / name
            if path.exists():
                st.image(str(path), caption=name, width='stretch')


def main() -> None:
    st.set_page_config(
        page_title="Stock Direction Classification (Research)",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(inject_styles(), unsafe_allow_html=True)

    metadata = load_json(str(MODELS / "metadata.json"))
    metrics = load_json(str(MODELS / "metrics.json"))

    if metadata is None or metrics is None:
        st.error(
            "Model artifacts not found. From the project root, run:\n\n"
            "`python run_pipeline.py`"
        )
        st.stop()

    try:
        featured = load_featured_frame()
    except Exception as exc:  # noqa: BLE001 — surface a clean UI message
        st.error(
            "Could not load the featured dataset for interactive charts. "
            "Ensure `data/stock_price_data.csv` exists or network access is available for kagglehub.\n\n"
            f"Details: {exc}"
        )
        st.stop()

    preds = load_predictions()

    render_hero()

    with st.sidebar:
        st.markdown("### Navigation")
        section = st.radio(
            "Go to section",
            options=[
                "Overview",
                "Market Data",
                "Model Evaluation",
                "Prediction Analysis",
                "Limitations",
            ],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.caption(
            f"**Series:** {metadata.get('ticker')} / {metadata.get('exchange')}\n\n"
            f"**Holdout:** {metadata.get('test_date_min')} → {metadata.get('test_date_max')}\n\n"
            f"**Test n:** {metadata.get('n_test')}"
        )
        st.markdown("---")
        st.caption(DISCLAIMER)

    if section == "Overview":
        render_overview(metadata, metrics, featured)
    elif section == "Market Data":
        render_market_data(featured)
    elif section == "Model Evaluation":
        render_model_evaluation(metrics)
    elif section == "Prediction Analysis":
        render_prediction_analysis(featured, preds)
    else:
        render_limitations()


if __name__ == "__main__":
    main()
