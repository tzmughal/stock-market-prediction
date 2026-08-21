# Stock Market Direction Prediction with Financial News Sentiment

## Project Overview

This project implements an end-to-end pipeline to predict the next-day price direction (up/down) of four technology stocks (AAPL, MSFT, NVDA, TSLA) using daily technical indicators and financial news sentiment.

### Notebooks:
1. `01_stock_data_collection.ipynb`: Downloads daily stock data using Yahoo Finance (`yfinance`).
2. `02_financial_news_collection.ipynb`: Collects and cleans tech stock news headlines from Hugging Face.
3. `03_sentiment_scoring_and_final_dataset.ipynb`: Scores sentiment of the headlines using FinBERT and merges it with the stock dataset.
4. `04_direction_prediction_model_v3.ipynb`: Trains, tunes, and evaluates Logistic Regression, Random Forest, and XGBoost classifiers.
5. `05_results_synthesis_and_discussion.ipynb`: Analyzes model performance, overfitting gaps, sentiment lead-lag correlation, and headline duplicates.

---

## Pipeline Architecture

```
[Stock Data (yfinance)] + [News Data (Hugging Face)]
                   │
                   ▼
       [FinBERT Sentiment Scoring]
                   │
                   ▼
         [Feature Engineering]
                   │
                   ▼
      [Direction Prediction Models]
                   │
                   ▼
        [Synthesis & Diagnostics]
```

---

## Summary of Findings

1. **Baseline vs. Trained Models:**
   None of the trained machine learning models (Logistic Regression, Random Forest, XGBoost) outperformed a naive majority-class baseline of predicting "Up". Walk-forward cross-validation accuracy for all models hovers around 50–51%.

2. **Diagnostics:**
   - **Overfitting:** XGBoost and Random Forest showed large training-to-test accuracy gaps (e.g., XGBoost achieved ~82% training accuracy but only 47.5% test accuracy), indicating significant overfitting.
   - **Sentiment Correlation:** Sentiment score correlates with same-day returns (~0.22) but has near-zero correlation with next-day returns (~ -0.02), indicating that sentiment is contemporaneous or lagging rather than leading.
   - **Headline Repetition:** Deduplication removed ~71% of raw headlines, which was caused by same-day wire duplication across different feeds rather than cross-date reposts.

---

## File Reference

- `notebooks/`: Contains the Jupyter notebooks for each step.
- `data/`:
  - `raw/`: Raw stock data and news headlines.
  - `processed/`: Sentiment scores and the combined daily dataset.
  - `models/`: Trained model binaries (`.joblib`) and test predictions.
  - `final/`: Walk-forward validation logs and summary tables.
