# How to Run

## Prerequisites
- Python 3.10+
- 8GB+ RAM

## Setup

1. **Virtual Environment:**
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

2. **Dependencies:**
```bash
pip install -r requirements.txt
```

## Running the Pipeline

Launch Jupyter:
```bash
jupyter notebook
```

Execute the notebooks sequentially:
1. `01_stock_data_collection.ipynb` (Downloads stock data)
2. `02_financial_news_collection.ipynb` (Downloads news headlines)
3. `03_sentiment_scoring_and_final_dataset.ipynb` (Scores sentiment using FinBERT; downloads ~440MB model weights on first run)
4. `04_direction_prediction_model_v3.ipynb` (Trains/tunes classifiers)
5. `05_results_synthesis_and_discussion.ipynb` (Saves final diagnostic results and summary tables)

Each step saves its outputs under the `data/` directory for the subsequent step.
