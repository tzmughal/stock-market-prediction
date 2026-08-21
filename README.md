# Stock Market Direction Prediction using Financial News Sentiment & Technical Analysis

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FinBERT](https://img.shields.io/badge/HuggingFace-FinBERT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/yiyanghkust/finbert)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-150458?style=for-the-badge)](https://xgboost.readthedocs.io/)

An end-to-end Machine Learning pipeline designed to predict the next-day price direction (Up / Down) of major technology stocks (**AAPL**, **MSFT**, **NVDA**, **TSLA**) by fusing daily technical indicators with financial news sentiment extracted via **FinBERT**.

---

## 🏗️ Pipeline Architecture

```
[Stock Price Data (yfinance)] + [Financial News Headlines (HuggingFace)]
                              │
                              ▼
                [FinBERT Sentiment Scoring (NLP)]
                              │
                              ▼
            [Feature Engineering & Feature Scaling]
                              │
                              ▼
       [Walk-Forward Cross-Validation & Model Training]
                              │
       (Logistic Regression | Random Forest | XGBoost)
                              │
                              ▼
               [Performance Evaluation & Diagnostics]
```

---

## 📈 Key Notebooks & Workflow

| Notebook | Purpose |
|---|---|
| `01_stock_data_collection.ipynb` | Downloads daily stock prices & technical indicators via Yahoo Finance (`yfinance`). |
| `02_financial_news_collection.ipynb` | Collects and cleans tech stock news headlines from financial news feeds. |
| `03_sentiment_scoring_and_final_dataset.ipynb` | Scores headline sentiment using **FinBERT** and merges sentiment features with price data. |
| `04_direction_prediction_model_v3.ipynb` | Trains, tunes, and evaluates Logistic Regression, Random Forest, and XGBoost classifiers using Walk-Forward Cross-Validation. |
| `05_results_synthesis_and_discussion.ipynb` | Diagnostic analysis of model generalization gaps, sentiment lead/lag correlation, and headline deduplication. |

---

## 🔬 Key Diagnostic Findings & Research Insights

1. **Walk-Forward Validation:** Evaluated using strict expanding-window time-series splits to prevent future data leakage.
2. **Lead vs. Lag Correlation:** Diagnostic correlation analysis revealed that news sentiment correlates strongly with *same-day* returns (~0.22) but shows near-zero predictive correlation with *next-day* returns (~ -0.02), highlighting that financial news sentiment is contemporaneous or lagging rather than leading.
3. **Headline Deduplication:** Removed ~71% duplicate headlines caused by same-day wire duplication across syndication feeds.

---

## 📁 Repository Structure

```
stock-market-prediction/
├── notebooks/                   # Jupyter notebooks (01 through 05)
├── data/
│   └── final/                  # Summary logs and evaluation comparison charts
├── documentation.md             # Project documentation details
├── HOW TO RUN.md                # Environment setup guide
├── requirements.txt             # Dependency file
└── README.md                    # Main portfolio README
```

*Note: Raw historical price CSVs, heavy headline datasets, and trained model binary artifacts (.joblib) are excluded from the public showcase repository.*

---

## 🛠️ Setup & Execution

```bash
git clone https://github.com/tzmughal/stock-market-prediction.git
cd stock-market-prediction
pip install -r requirements.txt
```

---

## 👤 Author
- **GitHub:** [@tzmughal](https://github.com/tzmughal)
