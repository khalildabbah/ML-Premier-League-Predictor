## EPL Match Prediction (XGBoost)

Machine Learning  English Premier League match outcomes predictor (home win / draw / away win) using historical data, engineered features (rolling form, odds, Elo, rest days), and a time-aware training pipeline.

### What problem this solves
- Aggregates seasonal CSVs into one clean dataset
- Builds robust features from raw match data
- Trains and evaluates a multiclass XGBoost model
- Simple workflow to refresh data as new games are played

---

## Key Features
- Robust CSV ingestion/merging from `data/`
- Automatic season detection from `YY-YY.csv` filenames
- Date parsing tolerant to multiple formats
- Feature engineering:
  - Rolling goals, shots, shots-on-target (window=5)
  - Betting odds → de-vigged implied probabilities for B365/PS/WH (if present)
  - Simple per-season Elo (pre-match ratings + diff)
  - Rest days for home/away teams
- Time-aware split: train (≤2018-19), validation (2019-20 to 2021-22), test (≥2022-23)
- Class weighting to help the Draw class
- Evaluation: accuracy, log loss, classification report, confusion matrix
- Recent matches prediction summary with hit rate

---

## Folder Structure

```
EPL-ML-PREDICTION/
├── data/                          # Season CSVs (input)
│   ├── 5-6.csv
│   ├── 6-7.csv
│   ├── 7-8.csv
│   ├── 8-9.csv
│   ├── 9-10.csv
│   ├── 10-11.csv
│   ├── 11-12.csv
│   ├── 12-13.csv
│   ├── 13-14.csv
│   ├── 14-15.csv
│   ├── 15-16.csv
│   ├── 16-17.csv
│   ├── 17-18.csv
│   ├── 18-19.csv
│   ├── 19-20.csv
│   ├── 20-21.csv
│   ├── 21-22.csv
│   ├── 22-23.csv
│   ├── 23-24.csv
│   ├── 24-25.csv
│   └── 25-26.csv
├── HOW_TO_UPDATE_DATA.md          # Step-by-step data update guide
├── main.py                        # Feature engineering, training, evaluation
├── update_data.py                 # Combine season CSVs into one dataset
├── premier_league_combined.csv    # Combined raw dataset (generated)
└── epl_with_features.csv          # Feature-enriched dataset (if generated)
```

---

## Setup

- Python: 3.9–3.11 recommended
- Install dependencies:

```bash
pip install -U pip
pip install pandas numpy scikit-learn xgboost
```

Optional (for exploration/plots/notebooks): `matplotlib seaborn jupyter`.

---

## Quickstart

```bash
# 1) Combine all season CSVs into one dataset
python update_data.py

# 2) Build features, train model, and evaluate
python main.py
```

You’ll see dataset sizes and features, test accuracy/log loss, a classification report, confusion matrix, the latest matches with predicted classes/confidence, and a hit rate summary.

---

## How to Update the Dataset

Add or refresh season files and rebuild the combined dataset.

- Place CSVs in `data/` named like `YY-YY.csv` (e.g., `25-26.csv`)
- CSVs should include at least: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG` (and `Div = "E0"` if present)
- The update script will:
  - Load all `data/*.csv`
  - Parse dates robustly
  - Filter to Premier League (`Div == "E0"`) when available
  - Deduplicate by `Season, Date, HomeTeam, AwayTeam` (prefers latest file)
  - Sort by season/date and write `premier_league_combined.csv`
  - Print latest matches and date range

Commands:

```bash
# Rebuild combined dataset
python update_data.py

# Re-train on updated data
python main.py
```

For a detailed walkthrough and data sources, see `HOW_TO_UPDATE_DATA.md`.

---

## Model and Approach

- Task: 3-class classification of match result
  - Labels: 0 = HomeWin, 1 = Draw, 2 = AwayWin
- Features (if present):
  - Rolling form (goals for/against, shots, shots-on-target; window=5)
  - Rest days for home/away
  - De-vigged implied probabilities from bookmaker odds (B365/PS/WH)
  - Elo per season (home/away pre-match, plus `elo_diff`), `K=20`
- Split:
  - Train ≤ 2018-19
  - Valid 2019-20 … 2021-22
  - Test ≥ 2022-23
- Class weights derived from training distribution (helps Draw)
- Model: XGBoost `XGBClassifier` (`objective=multi:softprob`, `num_class=3`, `random_state=42`)
  - Key params: `max_depth=6`, `learning_rate=0.05`, `n_estimators=3000`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `tree_method="hist"`

---

## Reproducibility

- Random seed: `RANDOM_SEED = 42` (and `random_state=42` for XGBoost)
- Time-aware splits by season prevent leakage
- Ensure consistent Python and library versions for identical results across machines

Minimal seed setup for custom additions:

```python
import numpy as np
import random
np.random.seed(42)
random.seed(42)
# XGBoost seed is set via random_state in main.py
```

---

## Roadmap / TODO
- Probability calibration (Platt/Isotonic) and Brier score
- Hyperparameter tuning (Optuna) and blocked cross-validation by season
- Save/load trained model and feature list
- Feature importance and SHAP-based explainability
- CLI flags for custom train/valid/test season ranges
- Backtesting simple betting strategies using predicted probabilities
- Dockerfile and requirements file for fully reproducible runs

---

## License

This project is intended for educational and personal use only.
Pull requests are welcome. Feel free to fork this repository and improve the project.

