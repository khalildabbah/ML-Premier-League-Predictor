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
        # Only calculate result if we have both FTHG and FTAG
        mask = df["FTHG"].notna() & df["FTAG"].notna()
        df["Result"] = np.nan
        df.loc[mask, "Result"] = (df.loc[mask, "FTHG"] < df.loc[mask, "FTAG"]).astype(int)*2 + (df.loc[mask, "FTHG"] == df.loc[mask, "FTAG"]).astype(int)
    return df


def add_rolling_form_goals(df: pd.DataFrame, window=5) -> pd.DataFrame:
    """Rolling goals for/against for home and away teams (last N games)."""
    df = df.sort_values(["Season", "Date"]).copy()
    for team_col, gf_col, ga_col, prefix in [
        ("HomeTeam","FTHG","FTAG","home"),
        ("AwayTeam","FTAG","FTHG","away")
    ]:
        # Only calculate for rows where we have goal data
        if gf_col in df.columns:
            df[f"{prefix}_gf_roll"] = (
                df.groupby(team_col, sort=False)[gf_col]
                  .rolling(window, min_periods=1).mean()
                  .reset_index(level=0, drop=True)
            )
        else:
            df[f"{prefix}_gf_roll"] = np.nan
            
        if ga_col in df.columns:
            df[f"{prefix}_ga_roll"] = (
                df.groupby(team_col, sort=False)[ga_col]
                  .rolling(window, min_periods=1).mean()
                  .reset_index(level=0, drop=True)
            )
        else:
            df[f"{prefix}_ga_roll"] = np.nan
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


def build_match_elo_single_season(sdf: pd.DataFrame, K=20, start_elo=1500, carry_forward_elo=None):
    """Compute pre-match Elo for each row within one season, then update after result.
    
    Args:
        sdf: DataFrame for one season
        K: Elo K-factor
        start_elo: Starting Elo for teams not seen before
        carry_forward_elo: Dict of {team: elo} to carry forward from previous season
    """
    sdf = sdf.sort_values("Date").copy()
    elo = carry_forward_elo.copy() if carry_forward_elo else {}
    preH, preA = [], []
    for _, r in sdf.iterrows():
        h, a = r["HomeTeam"], r["AwayTeam"]
        EH = elo.get(h, start_elo); EA = elo.get(a, start_elo)
        preH.append(EH); preA.append(EA)
        # expected home score
        expH = 1/(1+10**((EA-EH)/400))
        # actual result (only update if we have scores)
        if pd.notna(r.get("FTHG")) and pd.notna(r.get("FTAG")):
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
    return sdf, elo


def add_season_elo(df: pd.DataFrame, K=20, start_elo=1500) -> pd.DataFrame:
    """Compute Elo across all seasons, carrying forward from season to season."""
    df = df.sort_values(["Season","Date"]).copy()
    
    # Get unique seasons in order
    seasons = df["Season"].unique()
    seasons_sorted = sorted(seasons, key=lambda x: season_start_year(x) if isinstance(x, str) else -1)
    
    # Store Season column before groupby
    season_col = df["Season"].copy()
    elo_dict = {}  # Carries forward across seasons
    results = []
    
    for season in seasons_sorted:
        season_df = df[df["Season"] == season].copy()
        season_df, elo_dict = build_match_elo_single_season(season_df, K=K, start_elo=start_elo, carry_forward_elo=elo_dict)
        results.append(season_df)
    
    df = pd.concat(results, ignore_index=True)
    
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

# =========================
# PREDICT 2025-26 SEASON GAMES
# =========================
print("\n" + "="*100)
print("[PREDICTING 2025-26 SEASON GAMES]")
print("="*100)

# Load 2025-26 fixtures
future_csv = Path(r"C:\Users\khali\OneDrive\desktop\My-projects\EPL-ML-PREDICTION\data\25-26.csv")
if future_csv.exists():
    print(f"\n[LOADING] Loading 2025-26 fixtures from: {future_csv}")
    future_df = pd.read_csv(future_csv, low_memory=False)
    future_df.columns = future_df.columns.str.strip()
    
    # Ensure we have the required columns
    if all(c in future_df.columns for c in ["Date", "HomeTeam", "AwayTeam"]):
        # Add Season column if not present
        if "Season" not in future_df.columns:
            future_df["Season"] = "2025-26"
        
        # Parse date
        future_df["Date"] = parse_date_series(future_df["Date"])
        
        # Sort by date
        future_df = future_df.sort_values("Date").copy()
        
        print(f"Loaded {len(future_df)} fixtures for 2025-26 season")
        print(f"Date range: {future_df['Date'].min()} to {future_df['Date'].max()}")
        
        # Combine with historical data for feature engineering
        # We need historical data to calculate rolling stats and Elo
        all_data = pd.concat([df, future_df], ignore_index=True)
        all_data = all_data.sort_values(["Season", "Date"]).copy()
        
        # Add SeasonStartY for future games
        all_data["SeasonStartY"] = all_data["Season"].apply(season_start_year)
        
        # Calculate rolling stats for all data (including future games)
        print("\n[FEATURE ENGINEERING] Calculating features for 2025-26 games...")
        
        # 1) Rolling goals (for games with results, we can calculate; for future, use historical)
        all_data = add_rolling_form_goals(all_data, window=ROLL_WINDOW)
        
        # 2) Rolling shots
        all_data = add_rolling_form_shots(all_data, window=ROLL_WINDOW)
        
        # 3) Rest days
        all_data = add_rest_days(all_data)
        
        # 4) Odds (if available)
        all_data = add_odds_implied_probs(all_data)
        
        # 5) Elo (carries forward from previous seasons)
        print("Calculating Elo ratings (carrying forward from previous seasons)...")
        all_data = add_season_elo(all_data, K=ELO_K)
        
        # Extract just the 2025-26 games
        future_2025_26 = all_data[all_data["SeasonStartY"] == 2025].copy()
        
        print(f"\n[DEBUG] Future games after feature engineering: {len(future_2025_26)}")
        print(f"[DEBUG] Features expected: {features}")
        print(f"[DEBUG] Features available in future_2025_26: {[f for f in features if f in future_2025_26.columns]}")
        
        # Fill missing rolling stats for future games using most recent historical values
        # For each team, get their last known rolling stats from historical data
        historical_data = all_data[all_data["SeasonStartY"] < 2025].copy()
        
        if len(historical_data) > 0:
            # For each rolling stat feature, fill with team's most recent value
            # home_gf_roll and home_ga_roll use HomeTeam, away_gf_roll and away_ga_roll use AwayTeam
            rolling_features_map = {
                "home_gf_roll": "HomeTeam",
                "home_ga_roll": "HomeTeam", 
                "away_gf_roll": "AwayTeam",
                "away_ga_roll": "AwayTeam",
                "home_shots_roll": "HomeTeam",
                "away_shots_roll": "AwayTeam",
                "home_sot_roll": "HomeTeam",
                "away_sot_roll": "AwayTeam"
            }
            
            for feat, team_col in rolling_features_map.items():
                if feat in historical_data.columns:
                    # Get most recent non-null value for each team from historical data
                    # Sort by date and take last non-null value per team
                    hist_sorted = historical_data.sort_values("Date")
                    team_last_values = hist_sorted.groupby(team_col)[feat].apply(
                        lambda x: x.dropna().iloc[-1] if x.dropna().size > 0 else np.nan
                    ).to_dict()
                    
                    # Fill missing values in future games
                    mask = future_2025_26[feat].isna()
                    if mask.any():
                        future_2025_26.loc[mask, feat] = future_2025_26.loc[mask, team_col].map(team_last_values)
                        filled_count = future_2025_26.loc[mask, feat].notna().sum()
                        print(f"[DEBUG] Filled {filled_count}/{mask.sum()} missing values for {feat}")
        
        # Check which features are available
        available_features = [f for f in features if f in future_2025_26.columns]
        missing_features = [f for f in features if f not in future_2025_26.columns]
        
        if missing_features:
            print(f"\n[WARNING] Missing features for predictions: {missing_features}")
            print("These features will be filled with median values from training data.")
            # Fill missing features with median from training data
            for feat in missing_features:
                if feat in train_df.columns:
                    median_val = train_df[feat].median()
                    future_2025_26[feat] = median_val
                    available_features.append(feat)
                    print(f"[DEBUG] Filled {feat} with median: {median_val}")
        
        # Use only features that exist in both training and future data
        prediction_features = [f for f in features if f in available_features]
        
        if len(prediction_features) < len(features):
            print(f"\n[WARNING] Using {len(prediction_features)}/{len(features)} features for predictions")
        
        # Fill remaining NaN values with median from training data
        for feat in prediction_features:
            if future_2025_26[feat].isna().any():
                median_val = train_df[feat].median()
                future_2025_26[feat] = future_2025_26[feat].fillna(median_val)
                print(f"[DEBUG] Filled remaining NaN in {feat} with median: {median_val}")
        
        # Only drop rows with missing critical non-feature columns
        future_2025_26 = future_2025_26.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
        
        print(f"[DEBUG] Future games after filling and cleaning: {len(future_2025_26)}")
        print(f"[DEBUG] Missing values per feature:")
        for feat in prediction_features:
            missing_count = future_2025_26[feat].isna().sum()
            if missing_count > 0:
                print(f"  {feat}: {missing_count} missing")
        
        if len(future_2025_26) > 0:
            print(f"\n[PREDICTIONS] Making predictions for {len(future_2025_26)} games...")
            
            # Make predictions
            X_future = future_2025_26[prediction_features]
            y_pred_future = model.predict(X_future)
            y_prob_future = model.predict_proba(X_future)
            
            # Add predictions to dataframe
            future_2025_26["Predicted"] = y_pred_future
            future_2025_26["p_Home"] = y_prob_future[:, 0]
            future_2025_26["p_Draw"] = y_prob_future[:, 1]
            future_2025_26["p_Away"] = y_prob_future[:, 2]
            
            # Calculate actual results if available
            future_2025_26["Actual"] = np.nan
            mask_has_results = future_2025_26["FTHG"].notna() & future_2025_26["FTAG"].notna()
            if mask_has_results.any():
                future_2025_26.loc[mask_has_results, "Actual"] = (
                    (future_2025_26.loc[mask_has_results, "FTHG"] < future_2025_26.loc[mask_has_results, "FTAG"]).astype(int)*2 + 
                    (future_2025_26.loc[mask_has_results, "FTHG"] == future_2025_26.loc[mask_has_results, "FTAG"]).astype(int)
                )
            
            # Calculate success rate
            games_with_results = future_2025_26["Actual"].notna()
            if games_with_results.any():
                correct_predictions = (future_2025_26.loc[games_with_results, "Predicted"] == 
                                      future_2025_26.loc[games_with_results, "Actual"]).sum()
                total_with_results = games_with_results.sum()
                success_rate = (correct_predictions / total_with_results * 100) if total_with_results > 0 else 0
            else:
                success_rate = None
                correct_predictions = 0
                total_with_results = 0
            
            # Display summary
            print("\n" + "="*100)
            print("2025-26 SEASON PREDICTIONS SUMMARY")
            print("="*100)
            
            # Summary statistics
            home_wins = (y_pred_future == 0).sum()
            draws = (y_pred_future == 1).sum()
            away_wins = (y_pred_future == 2).sum()
            
            print(f"\n[PREDICTION SUMMARY]")
            print(f"Total Games Predicted: {len(future_2025_26)}")
            print(f"Predicted Home Wins: {home_wins} ({home_wins/len(future_2025_26)*100:.1f}%)")
            print(f"Predicted Draws: {draws} ({draws/len(future_2025_26)*100:.1f}%)")
            print(f"Predicted Away Wins: {away_wins} ({away_wins/len(future_2025_26)*100:.1f}%)")
            
            if success_rate is not None:
                print(f"\n[SUCCESS RATE]")
                print(f"Games with Results: {total_with_results}")
                print(f"Correct Predictions: {correct_predictions}/{total_with_results}")
                print(f"Success Rate: {success_rate:.1f}%")
            
            # Show sample of upcoming games (first 10 without results, or first 10 overall)
            upcoming_games = future_2025_26[~future_2025_26["Actual"].notna()].head(10)
            if len(upcoming_games) == 0:
                upcoming_games = future_2025_26.head(10)
            
            if len(upcoming_games) > 0:
                print(f"\n[UPCOMING GAMES - Sample of {len(upcoming_games)}]")
                print("-" * 100)
                for idx, row in upcoming_games.iterrows():
                    home_team = row["HomeTeam"]
                    away_team = row["AwayTeam"]
                    date_str = row["Date"].strftime("%Y-%m-%d") if pd.notna(row["Date"]) else "N/A"
                    
                    home_prob = row["p_Home"]
                    draw_prob = row["p_Draw"]
                    away_prob = row["p_Away"]
                    
                    # Most likely outcome
                    if home_prob >= draw_prob and home_prob >= away_prob:
                        prediction = f"{home_team} WIN"
                        confidence = home_prob
                    elif draw_prob >= home_prob and draw_prob >= away_prob:
                        prediction = "DRAW"
                        confidence = draw_prob
                    else:
                        prediction = f"{away_team} WIN"
                        confidence = away_prob
                    
                    print(f"{date_str} | {home_team:20s} vs {away_team:20s} | "
                          f"Pred: {prediction:25s} ({confidence:5.1%})")
                
                print("-" * 100)
            
            print("="*100)
            
            # Save predictions to CSV
            output_file = Path("predictions_2025_26.csv")
            output_cols = ["Date", "HomeTeam", "AwayTeam", "Predicted", "p_Home", "p_Draw", "p_Away"]
            if "FTHG" in future_2025_26.columns:
                output_cols.extend(["FTHG", "FTAG"])
            if "Actual" in future_2025_26.columns:
                output_cols.append("Actual")
            if "elo_home_pre" in future_2025_26.columns:
                output_cols.extend(["elo_home_pre", "elo_away_pre", "elo_diff"])
            
            # Only include columns that exist
            output_cols = [col for col in output_cols if col in future_2025_26.columns]
            future_2025_26[output_cols].to_csv(output_file, index=False)
            print(f"\n[SAVED] Predictions saved to: {output_file}")
        else:
            print("\n[ERROR] No valid games found for prediction after feature engineering.")
    else:
        print(f"\n[ERROR] Missing required columns in {future_csv}")
        print(f"Required: Date, HomeTeam, AwayTeam")
        print(f"Found: {list(future_df.columns)[:10]}")
else:
    print(f"\n[ERROR] Could not find 2025-26 fixtures file: {future_csv}")
    print("Please ensure the file exists in the data/ folder.")
