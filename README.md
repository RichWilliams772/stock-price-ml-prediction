# 📈 Stock Price Movement — Direction Classification

A machine learning research project that predicts the **next trading day's market direction (Up/Down)** using historical OHLC market data and technical indicators.

Rather than forecasting exact prices, this project formulates the problem as a **binary classification task**, demonstrating a complete end-to-end machine learning workflow including feature engineering, model training, evaluation, visualization, and an interactive Streamlit research dashboard.

> **Research Project**
> This project is intended for educational and research purposes only. It is **not financial advice** and should not be used for investment decisions.

---

# Dashboard Preview

<img width="1440" height="900" alt="Screenshot 2026-08-06 at 12 00 29 PM" src="https://github.com/user-attachments/assets/283bd9f0-5fb0-4ef8-937f-8184a95b68e1" />
<img width="1440" height="900" alt="Screenshot 2026-08-06 at 12 00 38 PM" src="https://github.com/user-attachments/assets/a50bed9a-87c7-490f-b873-f2606a65e9a3" />
<img width="1440" height="900" alt="Screenshot 2026-08-06 at 12 00 52 PM" src="https://github.com/user-attachments/assets/e584bfc4-1c05-419f-a2e8-6efb0787398e" />


---

# Project Overview

The objective is to determine whether the **next day's closing price will be higher or lower than today's closing price.**

The project demonstrates:

- Time-series preprocessing
- Feature engineering
- Chronological train/test splitting
- Machine learning classification
- Model comparison
- Performance evaluation
- Feature importance analysis
- Interactive visualization with Streamlit

Unlike many beginner stock prediction projects, this repository avoids random train/test splits that introduce data leakage and instead evaluates models using a proper chronological holdout.

---

# Features

✔ Historical OHLC price analysis

✔ Technical indicators

- 10-Day Moving Average
- 50-Day Moving Average
- Daily Returns

✔ Binary next-day direction prediction

✔ Multiple ML models

- Majority Class Baseline
- Logistic Regression
- Random Forest

✔ Chronological evaluation

✔ Confusion Matrix

✔ Feature Importance

✔ Rolling Accuracy Analysis

✔ Interactive Streamlit Dashboard

---

# Dataset

Market:

**CBX / XZAG**

Historical period:

2010-01-04 → 2015-12-30

After preprocessing:

| Dataset | Rows |
|---------|------|
| Training | 1,157 |
| Testing | 290 |
| Total | 1,447 |

Target Variable

```
1 = Next day's close > Today's close

0 = Otherwise
```

---

# Machine Learning Pipeline

```
Raw OHLC Data
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Chronological Train/Test Split
        │
        ▼
Model Training
        │
        ▼
Evaluation
        │
        ▼
Interactive Dashboard
```

---

# Models Evaluated

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|-----------|--------|-----|----------|
| Majority Baseline | 54.1% | — | — | — | — |
| Logistic Regression | 50.7% | 46.9% | 56.4% | 51.2% | 51.6% |
| Random Forest | 48.6% | 45.3% | 58.6% | 51.1% | 52.6% |

---

# Repository Structure

```text
stock-price-ml-prediction/

├── app/
│   ├── streamlit_app.py
│   └── components.py
│
├── src/
│   ├── data.py
│   ├── features.py
│   ├── train.py
│   ├── predict.py
│   └── evaluate.py
│
├── notebooks/
│   ├── stock_price_movement_analysis.ipynb
│   └── original_stock_price_prediction.ipynb
│
├── docs/
│   └── images/
│
├── tests/
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/RichWilliams772/stock-price-ml-prediction.git

cd stock-price-ml-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Run the machine learning pipeline

```bash
python run_pipeline.py
```

Launch the Streamlit dashboard

```bash
streamlit run app/streamlit_app.py
```

---

# Dashboard Sections

The dashboard includes:

- Overview
- Dataset Summary
- Market Data Exploration
- Model Evaluation
- Confusion Matrix
- Feature Importance
- Historical Prediction Explorer
- Rolling Accuracy
- Research Limitations

---

# Key Findings

- Random Forest achieved stronger recall than Logistic Regression but lower overall accuracy.
- A majority-class baseline outperformed both trained models in overall accuracy, highlighting the difficulty of predicting short-term market direction.
- Rolling accuracy fluctuated over time, illustrating changing market regimes.
- Feature importance was distributed across both price-based and technical indicators rather than dominated by a single variable.

These findings emphasize that financial time-series classification remains a challenging problem even with engineered technical features.

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Logistic Regression
- Plotly
- Matplotlib
- Streamlit
- Jupyter Notebook

---

# Future Improvements

- XGBoost
- LightGBM
- LSTM / GRU
- Transformer models
- Walk-forward validation
- Hyperparameter optimization
- Additional technical indicators
- Sentiment analysis
- Macro-economic features

---

# Disclaimer

This repository is intended for educational and research purposes only.

It is **not** financial advice, investment guidance, or a trading system. Past market performance does not guarantee future results.

---

# Author

**Richelle Williams**

MS Data Science & Analytics

PhD Student in Electrical Engineering & Computer Science

Artificial Intelligence • Machine Learning • Data Science

GitHub:
https://github.com/RichWilliams772
