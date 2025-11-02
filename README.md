ML Pipeline for Trading Bot (POC_Safem0de_IS)

This workspace contains example notebooks and a new pipeline notebook `ML_Pipeline_for_Bot_Trade.ipynb`.

Files created:

- ML_Pipeline_for_Bot_Trade.ipynb  — A runnable skeleton pipeline: data load (from `data/`), EDA, feature engineering, labeling, RandomForest baseline, and a simple backtester.\n
- requirements.txt  — minimal Python dependencies.\n

How to run

1. Create a Python environment (recommended):

   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

2. Launch Jupyter in the workspace root and open the notebook:

   jupyter lab

3. In the notebook, update `DATA_PATH` in the first code cell if you want to use a different CSV from `data/`.

Notes & next steps

- The notebook is intentionally minimal to be a starting point. After initial experiments, consider adding LightGBM, walk-forward CV, hyperparameter tuning (Optuna), and a more realistic backtester (e.g., zipline/backtesting.py/vectorbt) with transaction costs and position sizing.
- The `POC_*` notebooks already contain many useful helpers for candlestick patterns, UMAP/PCA/cluster analysis, and LSTM/CNN models; you can re-use feature engineering and model definitions from them.
