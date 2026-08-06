"""UI helpers for the research Streamlit dashboard (presentation only)."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Neutral financial-analytics palette (not brokerage branding).
COLORS = {
    "navy": "#1e3a5f",
    "slate": "#475569",
    "gray": "#64748b",
    "muted": "#94a3b8",
    "border": "#e2e8f0",
    "bg_card": "#f8fafc",
    "bg_soft": "#f1f5f9",
    "accent": "#3b82f6",
    "caution": "#b45309",
    "caution_bg": "#fffbeb",
    "down": "#7f1d1d",
    "up": "#14532d",
    "correct": "#0f766e",
    "incorrect": "#b91c1c",
    "threshold": "#64748b",
}

FEATURE_LABELS = {
    "last_value": "Closing Price",
    "open_value": "Opening Price",
    "high_value": "Daily High",
    "low_value": "Daily Low",
    "Return": "Daily Return",
    "MA10": "10-Day Moving Average",
    "MA50": "50-Day Moving Average",
}

DISCLAIMER = (
    "This project is for educational and research purposes only. "
    "It is not financial advice and should not be used to make investment decisions."
)

PLOTLY_LAYOUT = dict(
    font=dict(family="Source Sans 3, IBM Plex Sans, Segoe UI, sans-serif", size=13, color="#334155"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="#ffffff",
    margin=dict(l=48, r=24, t=56, b=48),
    hovermode="x unified",
)


def fmt_pct(value: Optional[float], digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.{digits}f}%"


def fmt_int(value: Any) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def readable_feature(name: str) -> str:
    return FEATURE_LABELS.get(name, name)


def inject_styles() -> str:
    return """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

      html, body, [class*="css"] {
        font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      }
      .block-container {
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1180px;
      }
      #MainMenu {visibility: hidden;}
      footer {visibility: hidden;}

      .hero-wrap {
        background: linear-gradient(135deg, #f8fafc 0%, #eef2f7 55%, #e8eef6 100%);
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 1.6rem 1.8rem 1.35rem 1.8rem;
        margin-bottom: 1.1rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      .hero-kicker {
        color: #64748b;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
      }
      .hero-title {
        font-family: "Source Serif 4", Georgia, serif;
        font-size: 2.05rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.2;
        margin: 0 0 0.55rem 0;
      }
      .hero-subtitle {
        color: #475569;
        font-size: 1.02rem;
        line-height: 1.55;
        max-width: 46rem;
        margin: 0 0 0.9rem 0;
      }
      .disclaimer-banner {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-left: 4px solid #c2410c;
        border-radius: 10px;
        padding: 0.75rem 0.95rem;
        color: #7c2d12;
        font-size: 0.92rem;
        line-height: 1.45;
      }

      .section-label {
        color: #64748b;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin: 0.35rem 0 0.65rem 0;
      }
      .section-title {
        font-family: "Source Serif 4", Georgia, serif;
        font-size: 1.45rem;
        color: #0f172a;
        margin: 0 0 0.35rem 0;
      }
      .section-note {
        color: #64748b;
        font-size: 0.92rem;
        margin-bottom: 0.85rem;
      }

      .kpi-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 0.85rem;
      }
      @media (max-width: 900px) {
        .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      }
      @media (max-width: 560px) {
        .kpi-grid { grid-template-columns: 1fr; }
      }
      .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.85rem 0.95rem;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        min-height: 84px;
      }
      .kpi-card.caution {
        background: #fffbeb;
        border-color: #fcd34d;
      }
      .kpi-card.neutral-accent {
        border-top: 3px solid #3b82f6;
      }
      .kpi-label {
        color: #64748b;
        font-size: 0.74rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
      }
      .kpi-value {
        color: #0f172a;
        font-size: 1.28rem;
        font-weight: 650;
        line-height: 1.25;
      }
      .kpi-hint {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 0.25rem;
      }

      .panel {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        margin: 0.7rem 0 1rem 0;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }
      .panel.caution {
        background: #fffbeb;
        border-color: #fcd34d;
      }
      .panel h4 {
        margin: 0 0 0.45rem 0;
        color: #0f172a;
        font-size: 1rem;
      }
      .panel p, .panel li {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.5;
      }
      .panel ul { margin: 0.2rem 0 0 1.1rem; padding: 0; }

      .cm-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        margin: 0.6rem 0 0.9rem 0;
      }
      .cm-cell {
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        padding: 0.9rem 1rem;
        background: #f8fafc;
      }
      .cm-cell.good { background: #ecfdf5; border-color: #a7f3d0; }
      .cm-cell.bad { background: #fef2f2; border-color: #fecaca; }
      .cm-name { font-size: 0.78rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.04em; }
      .cm-value { font-size: 1.55rem; font-weight: 700; color: #0f172a; margin: 0.2rem 0; }
      .cm-desc { font-size: 0.82rem; color: #64748b; line-height: 1.35; }

      .tag {
        display: inline-block;
        border-radius: 999px;
        padding: 0.15rem 0.55rem;
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-left: 0.35rem;
        vertical-align: middle;
      }
      .tag.primary { background: #e0e7ff; color: #3730a3; }
      .tag.baseline { background: #e2e8f0; color: #334155; }
      .tag.best { background: #ecfeff; color: #155e75; }

      .takeaway-list li { margin-bottom: 0.35rem; }
    </style>
    """


def kpi_card_html(
    label: str,
    value: str,
    hint: str = "",
    *,
    caution: bool = False,
    accent: bool = False,
) -> str:
    classes = ["kpi-card"]
    if caution:
        classes.append("caution")
    if accent:
        classes.append("neutral-accent")
    hint_html = f'<div class="kpi-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="{" ".join(classes)}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{hint_html}"
        f"</div>"
    )


def kpi_row(cards: list[str]) -> str:
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def section_header(title: str, note: str = "", label: str = "") -> str:
    label_html = f'<div class="section-label">{label}</div>' if label else ""
    note_html = f'<div class="section-note">{note}</div>' if note else ""
    return f'{label_html}<div class="section-title">{title}</div>{note_html}'


def apply_plotly_theme(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT, height=height)
    fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
    return fig


def chart_price_history(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["last_value"],
            mode="lines",
            name="Closing price",
            line=dict(color=COLORS["navy"], width=1.8),
        )
    )
    fig.update_layout(
        title="CBX Closing Price History",
        xaxis_title="Date",
        yaxis_title="Last value",
        showlegend=False,
    )
    return apply_plotly_theme(fig, height=400)


def chart_moving_averages(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=df["date"], y=df["last_value"], name="Close", line=dict(color=COLORS["navy"], width=1.5))
    )
    if "MA10" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["MA10"], name="MA10", line=dict(color="#c2410c", width=1.3))
        )
    if "MA50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df["date"], y=df["MA50"], name="MA50", line=dict(color="#0f766e", width=1.3))
        )
    fig.update_layout(title="Price and Trailing Moving Averages", xaxis_title="Date", yaxis_title="Price")
    return apply_plotly_theme(fig, height=400)


def chart_daily_returns(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["Return"],
            mode="lines",
            name="Daily return",
            line=dict(color=COLORS["slate"], width=1.0),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLORS["muted"])
    fig.update_layout(title="Daily Return", xaxis_title="Date", yaxis_title="Return", showlegend=False)
    return apply_plotly_theme(fig, height=320)


def chart_class_distribution(y: pd.Series) -> go.Figure:
    counts = y.value_counts().sort_index()
    labels = ["Down (0)", "Up (1)"]
    values = [int(counts.get(0, 0)), int(counts.get(1, 0))]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=[COLORS["down"], COLORS["up"]],
                text=values,
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        title="Target Class Distribution (Next-Day Direction)",
        yaxis_title="Count",
        xaxis_title="",
        showlegend=False,
    )
    return apply_plotly_theme(fig, height=360)


def chart_model_comparison(metrics: dict[str, Any]) -> go.Figure:
    names = ["majority_class", "logistic_regression", "random_forest"]
    labels = ["Majority Class\n(baseline)", "Logistic Regression", "Random Forest\n(primary)"]
    metric_keys = ["accuracy", "precision", "recall", "f1"]
    metric_names = ["Accuracy", "Precision", "Recall", "F1"]
    palette = ["#64748b", "#1e3a5f", "#3b82f6", "#0f766e"]

    fig = go.Figure()
    x = np.arange(len(labels))
    width = 0.18
    for i, (key, mname, color) in enumerate(zip(metric_keys, metric_names, palette)):
        vals = [100.0 * float(metrics[n].get(key) or 0.0) for n in names]
        fig.add_trace(
            go.Bar(
                name=mname,
                x=labels,
                y=vals,
                marker_color=color,
                text=[f"{v:.1f}%" for v in vals],
                textposition="outside",
                offsetgroup=str(i),
            )
        )
    fig.update_layout(
        title="Model Comparison on Chronological Test Set",
        yaxis_title="Score (%)",
        yaxis=dict(range=[0, 110]),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return apply_plotly_theme(fig, height=420)


def chart_confusion_matrix(cm: list[list[int]]) -> go.Figure:
    matrix = np.asarray(cm)
    z = matrix
    text = np.array(
        [
            [f"True Down<br>{matrix[0, 0]}", f"False Up<br>{matrix[0, 1]}"],
            [f"False Down<br>{matrix[1, 0]}", f"True Up<br>{matrix[1, 1]}"],
        ]
    )
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Predicted Down", "Predicted Up"],
            y=["Actual Down", "Actual Up"],
            text=text,
            texttemplate="%{text}",
            colorscale=[
                [0.0, "#f8fafc"],
                [0.5, "#bfdbfe"],
                [1.0, "#1e3a5f"],
            ],
            showscale=True,
            colorbar=dict(title="Count"),
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        )
    )
    fig.update_layout(title="Random Forest Confusion Matrix (Test Set)")
    fig.update_yaxes(autorange="reversed")
    return apply_plotly_theme(fig, height=420)


def chart_feature_importance(records: list[dict]) -> go.Figure:
    fi = pd.DataFrame(records).copy()
    fi["Label"] = fi["Feature"].map(readable_feature)
    fi = fi.sort_values("Importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=fi["Importance"],
            y=fi["Label"],
            orientation="h",
            marker_color=COLORS["navy"],
            text=[f"{v:.3f}" for v in fi["Importance"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title="Top Predictive Features (Random Forest)",
        xaxis_title="Importance",
        yaxis_title="",
        showlegend=False,
    )
    return apply_plotly_theme(fig, height=400)


def chart_prediction_analysis(preds: pd.DataFrame, window: int = 30) -> go.Figure:
    df = preds.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["correct"] = df["actual"] == df["predicted"]
    rolling = df["correct"].astype(float).rolling(window=window, min_periods=max(5, window // 5)).mean()

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.62, 0.38],
        vertical_spacing=0.08,
        subplot_titles=(
            "Predicted Probability of Up (Test Period)",
            f"Rolling Directional Accuracy ({window}-day window)",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["probability_up"],
            mode="lines",
            name="P(Up)",
            line=dict(color=COLORS["accent"], width=1.6),
        ),
        row=1,
        col=1,
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color=COLORS["threshold"], row=1, col=1)

    actual_up = df[df["actual"] == 1]
    actual_down = df[df["actual"] == 0]
    fig.add_trace(
        go.Scatter(
            x=actual_up["date"],
            y=actual_up["probability_up"],
            mode="markers",
            name="Actual Up",
            marker=dict(symbol="triangle-up", size=8, color=COLORS["up"], opacity=0.75),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=actual_down["date"],
            y=actual_down["probability_up"],
            mode="markers",
            name="Actual Down",
            marker=dict(symbol="triangle-down", size=8, color=COLORS["down"], opacity=0.75),
        ),
        row=1,
        col=1,
    )

    correct = df[df["correct"]]
    incorrect = df[~df["correct"]]
    fig.add_trace(
        go.Scatter(
            x=correct["date"],
            y=[1.03] * len(correct),
            mode="markers",
            name="Correct",
            marker=dict(symbol="circle", size=6, color=COLORS["correct"]),
            hovertemplate="Correct on %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=incorrect["date"],
            y=[1.06] * len(incorrect),
            mode="markers",
            name="Incorrect",
            marker=dict(symbol="x", size=7, color=COLORS["incorrect"]),
            hovertemplate="Incorrect on %{x}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=rolling,
            mode="lines",
            name="Rolling accuracy",
            line=dict(color=COLORS["navy"], width=1.8),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0.5, line_dash="dot", line_color=COLORS["muted"], row=2, col=1)

    fig.update_yaxes(title_text="P(Up)", range=[-0.05, 1.12], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0))
    return apply_plotly_theme(fig, height=560)


def build_comparison_table(metrics: dict[str, Any]) -> pd.DataFrame:
    rows = []
    mapping = [
        ("majority_class", "Majority Class (baseline)"),
        ("logistic_regression", "Logistic Regression"),
        ("random_forest", "Random Forest (primary research model)"),
    ]
    for key, label in mapping:
        m = metrics[key]
        rows.append(
            {
                "Model": label,
                "Accuracy": m.get("accuracy"),
                "Precision": m.get("precision"),
                "Recall": m.get("recall"),
                "F1": m.get("f1"),
                "ROC-AUC": m.get("roc_auc"),
            }
        )
    return pd.DataFrame(rows)


def comparison_table_for_display(metrics: dict[str, Any]) -> pd.DataFrame:
    """Return a display table with percentage strings and best-in-column markers."""
    raw = build_comparison_table(metrics)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    best = {}
    for col in metric_cols:
        numeric = pd.to_numeric(raw[col], errors="coerce")
        if numeric.notna().any():
            best[col] = numeric.idxmax()
        else:
            best[col] = None

    display_rows = []
    for idx, row in raw.iterrows():
        out = {"Model": row["Model"]}
        for col in metric_cols:
            val = row[col]
            text = fmt_pct(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else "—"
            if best.get(col) == idx and text != "—":
                text = f"{text} ★"
            out[col] = text
        display_rows.append(out)
    return pd.DataFrame(display_rows)


def highlight_best_metrics(df: pd.DataFrame) -> "pd.io.formats.style.Styler":
    """Highlight the strongest numeric value in each metric column."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

    def _style_col(col: pd.Series):
        styles = [""] * len(col)
        numeric = pd.to_numeric(col, errors="coerce")
        if numeric.notna().sum() == 0:
            return styles
        best_idx = numeric.idxmax()
        for i, idx in enumerate(col.index):
            if idx == best_idx and pd.notna(numeric.loc[idx]):
                styles[i] = "background-color: #ecfeff; font-weight: 600;"
        return styles

    styler = df.style.format(
        {
            "Accuracy": lambda v: fmt_pct(v),
            "Precision": lambda v: fmt_pct(v),
            "Recall": lambda v: fmt_pct(v),
            "F1": lambda v: fmt_pct(v),
            "ROC-AUC": lambda v: fmt_pct(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else "—",
        }
    )
    for col in metric_cols:
        styler = styler.apply(_style_col, subset=[col])
    return styler


def positive_class_share(featured: pd.DataFrame) -> float:
    return float(featured["Target"].mean())
