# EPL Match Prediction with XGBoost
# End-to-end EPL predictor:
# - load combined CSV
# - clean dtypes & dates (robust parser)
# - features: rolling form (goals, shots, shots OT), odds -> implied probs, simple Elo, rest days
# - time-aware split (train/valid/test)
# - train XGBoost with early stopping & class weights
# - evaluate

import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
import xgboost as xgb

# =========================
# CONFIG
# =========================
# Load directly from combined CSV - this script handles all feature engineering
CSV_PATH = Path(r"C:\Users\khali\OneDrive\desktop\My-projects\EPL-ML-PREDICTION\premier_league_combined.csv")

ROLL_WINDOW = 5     # rolling window for form
ELO_K       = 20    # Elo K-factor
RANDOM_SEED = 42

# Seasons split (by start year, e.g., "2019-20" -> 2019)
TRAIN_END_Y = 2018     # train <= 2018-19
VALID_END_Y = 2021     # valid: 2019-20 .. 2021-22
# test: >= 2022-23


# =========================
# UTILITIES
# =========================
def season_start_year(s: str) -> int:
    """Extract season start year from strings like '2005-06', '2019-20'."""
    if not isinstance(s, str):
        return -1
    m = re.search(r"(\d{4})", s)
    if m:
        return int(m.group(1))
    m2 = re.search(r"^(\d{2})-", s)
    if m2:
        yy = int(m2.group(1))
        return 2000 + yy if yy < 50 else 1900 + yy
    return -1


def parse_date_series(s: pd.Series) -> pd.Series:
    """Robust date parse:
       - If looks like YYYY-MM-DD -> dayfirst=False
       - Else assume European style -> dayfirst=True
    """
    s = s.astype(str)
    mask_iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    out = pd.to_datetime(s.where(mask_iso), format="%Y-%m-%d", errors="coerce")
    out2 = pd.to_datetime(s.where(~mask_iso), dayfirst=True, errors="coerce")
    out = out.fillna(out2)
    return out


def to_numeric_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_result_column(df: pd.DataFrame) -> pd.DataFrame:
    # 0=HomeWin, 1=Draw, 2=AwayWin
    if "Result" not in df.columns:
        df["Result"] = (df["FTHG"] < df["FTAG"]).astype(int)*2 + (df["FTHG"] == df["FTAG"]).astype(int)
    return df


def add_rolling_form_goals(df: pd.DataFrame, window=5) -> pd.DataFrame:
    """Rolling goals for/against for home and away teams (last N games)."""
    df = df.sort_values(["Season", "Date"]).copy()
    for team_col, gf_col, ga_col, prefix in [
        ("HomeTeam","FTHG","FTAG","home"),
        ("AwayTeam","FTAG","FTHG","away")
    ]:
        df[f"{prefix}_gf_roll"] = (
            df.groupby(team_col, sort=False)[gf_col]
              .rolling(window, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
        df[f"{prefix}_ga_roll"] = (
            df.groupby(team_col, sort=False)[ga_col]
              .rolling(window, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
    return df


def add_rolling_form_shots(df: pd.DataFrame, window=5) -> pd.DataFrame:
    """Rolling shots and shots-on-target for both sides."""
    df = df.sort_values(["Season", "Date"]).copy()
    # Shots
    if "HS" in df.columns and "AS" in df.columns:
        df["home_shots_roll"] = (
            df.groupby("HomeTeam", sort=False)["HS"]
              .rolling(window, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
        df["away_shots_roll"] = (
            df.groupby("AwayTeam", sort=False)["AS"]
              .rolling(window, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
    # Shots on target
    if "HST" in df.columns and "AST" in df.columns:
        df["home_sot_roll"] = (
            df.groupby("HomeTeam", sort=False)["HST"]
              .rolling(window, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
        df["away_sot_roll"] = (
            df.groupby("AwayTeam", sort=False)["AST"]
              .rolling(window, min_periods=1).mean()
              .reset_index(level=0, drop=True)
        )
    return df


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest days (days since last match) for home & away teams, reset each season."""
    def _per_season(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("Date").copy()
        last_played = {}
        home_rest, away_rest = [], []
        for _, r in g.iterrows():
            h, a = r["HomeTeam"], r["AwayTeam"]
            d = r["Date"]
            home_rest.append((d - last_played[h]).days if h in last_played else np.nan)
            away_rest.append((d - last_played[a]).days if a in last_played else np.nan)
            last_played[h] = d
            last_played[a] = d
        g["home_rest_days"] = home_rest
        g["away_rest_days"] = away_rest
        return g

    # Store Season column before groupby
    season_col = df["Season"].copy()
    
    try:
        # pandas >=2.2 supports include_groups
        df = df.groupby("Season", group_keys=False).apply(_per_season, include_groups=False)
    except TypeError:
        # fallback for older pandas
        df = df.groupby("Season", group_keys=False).apply(_per_season)
    
    # Re-add Season column if it was lost
    if "Season" not in df.columns:
        df["Season"] = season_col
    
    return df


def add_odds_implied_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Convert odds to implied probs (de-vigged) for common books if present."""
    for prefix in ["B365", "PS", "WH"]:
        H, D, A = f"{prefix}H", f"{prefix}D", f"{prefix}A"
        if all(col in df.columns for col in [H, D, A]):
            to_numeric_cols(df, [H, D, A])
            oH, oD, oA = df[H], df[D], df[A]
            pH_raw, pD_raw, pA_raw = 1/oH, 1/oD, 1/oA
            overround = pH_raw + pD_raw + pA_raw
            df[f"{prefix}_pH"] = pH_raw / overround
            df[f"{prefix}_pD"] = pD_raw / overround
            df[f"{prefix}_pA"] = pA_raw / overround
            df[f"{prefix}_margin"] = overround - 1.0
    return df


def build_match_elo_single_season(sdf: pd.DataFrame, K=20, start_elo=1500) -> pd.DataFrame:
    """Compute pre-match Elo for each row within one season, then update after result."""
    sdf = sdf.sort_values("Date").copy()
    elo = {}
    preH, preA = [], []
    for _, r in sdf.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        EH = elo.get(h, start_elo); EA = elo.get(a, start_elo)
        preH.append(EH); preA.append(EA)
        # expected home score
        expH = 1/(1+10**((EA-EH)/400))
        # actual result
        if r["FTHG"] > r["FTAG"]: RH = 1.0
        elif r["FTHG"] == r["FTAG"]: RH = 0.5
        else: RH = 0.0
        RA = 1.0 - RH
        # update
        elo[h] = EH + K*(RH - expH)
        elo[a] = EA + K*(RA - (1-expH))
    sdf["elo_home_pre"] = preH
    sdf["elo_away_pre"] = preA
    sdf["elo_diff"] = sdf["elo_home_pre"] - sdf["elo_away_pre"]
    return sdf


def add_season_elo(df: pd.DataFrame, K=20) -> pd.DataFrame:
    df = df.sort_values(["Season","Date"]).copy()
    
    # Store Season column before groupby
    season_col = df["Season"].copy()
    
    try:
        df = df.groupby("Season", group_keys=False).apply(build_match_elo_single_season, K=K, include_groups=False)
    except TypeError:
        df = df.groupby("Season", group_keys=False).apply(build_match_elo_single_season, K=K)
    
    # Re-add Season column if it was lost
    if "Season" not in df.columns:
        df["Season"] = season_col
    
    return df


# =========================
# LOAD & CLEAN
# =========================
print("[LOADING] Loading:", CSV_PATH)
df = pd.read_csv(CSV_PATH, low_memory=False)
df.columns = df.columns.str.strip()

# Ensure core columns exist
needed = ["Season","Date","HomeTeam","AwayTeam","FTHG","FTAG"]
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}. Columns present: {list(df.columns)[:30]} ...")

# Parse date (robust, no warnings)
df["Date"] = parse_date_series(df["Date"])

# Coerce common numerics
to_numeric_cols(df, ["FTHG","FTAG","HTHG","HTAG","HS","AS","HST","AST"])

# Result label
df = add_result_column(df)

# =========================
# FEATURES
# =========================
# 1) Rolling goals (if not already present)
if not all(c in df.columns for c in ["home_gf_roll","home_ga_roll","away_gf_roll","away_ga_roll"]):
    df = add_rolling_form_goals(df, window=ROLL_WINDOW)

# 2) Rolling shots / shots on target
df = add_rolling_form_shots(df, window=ROLL_WINDOW)

# 3) Rest days per team (reset each season)
df = add_rest_days(df)

# 4) Odds -> implied probabilities
print("Before odds processing - Season column exists:", 'Season' in df.columns)
df = add_odds_implied_probs(df)
print("After odds processing - Season column exists:", 'Season' in df.columns)

# 5) Elo
if "elo_diff" not in df.columns:
    print("Adding Elo features...")
    print("DataFrame shape before Elo:", df.shape)
    print("Columns before Elo:", list(df.columns)[:10])
    df = add_season_elo(df, K=ELO_K)

# =========================
# SPLIT
# =========================
df["SeasonStartY"] = df["Season"].apply(season_start_year)

train_df = df[df["SeasonStartY"] <= TRAIN_END_Y]
valid_df = df[(df["SeasonStartY"] > TRAIN_END_Y) & (df["SeasonStartY"] <= VALID_END_Y)]
test_df  = df[df["SeasonStartY"] >  VALID_END_Y]

# Choose features that exist
candidate_features = [
    # goals form
    "home_gf_roll","home_ga_roll","away_gf_roll","away_ga_roll",
    # shots form
    "home_shots_roll","away_shots_roll","home_sot_roll","away_sot_roll",
    # rest days
    "home_rest_days","away_rest_days",
    # odds implied
    "B365_pH","B365_pD","B365_pA",
    "PS_pH","PS_pD","PS_pA",
    "WH_pH","WH_pD","WH_pA",
    # Elo
    "elo_diff","elo_home_pre","elo_away_pre"
]
features = [c for c in candidate_features if c in df.columns]

if len(features) == 0:
    raise ValueError("No features found. Check that rolling/odds/elo features exist in your CSV.")

dropna_cols = features + ["Result","Date"]
train_df = train_df.dropna(subset=dropna_cols)
valid_df = valid_df.dropna(subset=dropna_cols)
test_df  = test_df.dropna(subset=dropna_cols)

X_train, y_train = train_df[features], train_df["Result"]
X_valid, y_valid = valid_df[features], valid_df["Result"]
X_test,  y_test  = test_df[features],  test_df["Result"]

print(f"\n[DATASETS] Train: {X_train.shape[0]} | Valid: {X_valid.shape[0]} | Test: {X_test.shape[0]}")
print("Features used:", features)

# =========================
# CLASS WEIGHTS (help the Draw class)
# =========================
cls_counts = y_train.value_counts().reindex([0,1,2]).fillna(0)
total = cls_counts.sum()
weights = {c: (total/(3*cnt)) if cnt>0 else 1.0 for c,cnt in cls_counts.items()}
w_train = y_train.map(weights)

# =========================
# TRAIN XGBOOST (early stopping)
# =========================
model = xgb.XGBClassifier(
    objective="multi:softprob",
    num_class=3,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=3000,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=RANDOM_SEED,
    tree_method="hist",
    eval_metric="mlogloss"
)

model.fit(
    X_train, y_train,
    sample_weight=w_train,
    eval_set=[(X_valid, y_valid)],
    verbose=False,

)

# =========================
# EVALUATE
# =========================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)
ll  = log_loss(y_test, y_prob)

print("\n[EVALUATION] Results on TEST set")
print("Accuracy:", round(acc, 4))
print("LogLoss :", round(ll, 4))
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["HomeWin","Draw","AwayWin"]))

cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])
print("Confusion matrix (rows=true, cols=pred):\n", cm)

# =========================
# MOST RECENT GAMES ANALYSIS
# =========================
# Get most recent games from entire dataset that have all features AND results
core_cols = ["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "Result"]
recent_full = df.dropna(subset=core_cols + features).copy()
recent_full = recent_full.sort_values(["SeasonStartY", "Date"], ascending=[False, False])

# Show 30 most recent games that have all features
num_recent = 30
recent = recent_full.head(num_recent).copy()

# Make predictions
prob_cols = ["p_Home", "p_Draw", "p_Away"]
recent_probs = model.predict_proba(recent[features])
recent[prob_cols] = recent_probs

print(f"\n[MOST RECENT GAMES] Showing {num_recent} most recent matches:")
print("=" * 100)

correct_predictions = 0
total_predictions = len(recent)

for idx, row in recent.iterrows():
    home_team = row["HomeTeam"]
    away_team = row["AwayTeam"]
    date_str = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "N/A"
    
    home_prob = row["p_Home"]
    draw_prob = row["p_Draw"]
    away_prob = row["p_Away"]
    
    # Most likely outcome
    if home_prob >= draw_prob and home_prob >= away_prob:
        predicted_class = 0
        prediction = f"{home_team} WIN"
        confidence = home_prob
    elif draw_prob >= home_prob and draw_prob >= away_prob:
        predicted_class = 1
        prediction = "DRAW"
        confidence = draw_prob
    else:
        predicted_class = 2
        prediction = f"{away_team} WIN"
        confidence = away_prob
    
    # Actual result
    actual_class = int(row["Result"])
    if actual_class == 0:
        actual = f"{home_team} WIN"
    elif actual_class == 1:
        actual = "DRAW"
    else:
        actual = f"{away_team} WIN"
    
    # Check if correct
    is_correct = predicted_class == actual_class
    if is_correct:
        correct_predictions += 1
        status = "[CORRECT]"
    else:
        status = "[WRONG]"
    
    # Format score
    score = f"{int(row['FTHG'])}-{int(row['FTAG'])}"
    
    print(f"{date_str} | {home_team:20s} vs {away_team:20s} | Score: {score:5s} | "
          f"Pred: {prediction:20s} ({confidence:5.1%}) | Actual: {actual:20s} | {status}")

print("=" * 100)

# Calculate hit rate
total_predictions = len(recent)
hit_rate = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

print(f"\n[HIT RATE SUMMARY]")
print(f"Games Analyzed: {total_predictions}")
print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
print(f"Hit Rate: {hit_rate:.1f}%")
print(f"Date Range: {recent['Date'].min().strftime('%Y-%m-%d')} to {recent['Date'].max().strftime('%Y-%m-%d')}")

print("\n[DONE] Training complete!")
