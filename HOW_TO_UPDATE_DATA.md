# How to Update Your EPL Dataset with New Games

This guide explains how to add newer Premier League game data to improve your predictions.

## Quick Start

1. **Add new CSV files** to the `data/` folder (see below for where to get them)
2. **Run the update script:**
   ```bash
   python update_data.py
   ```
3. **Train the model (this adds all features and trains):**
   ```bash
   python main.py
   ```

**That's it!** `main.py` handles all feature engineering and model training automatically.

## Detailed Steps

### Step 1: Get New Data Files

You need CSV files with Premier League match data. Common sources:

- **Football-Data.co.uk**: https://www.football-data.co.uk/englandm.php
  - Download the latest season CSV file
  - Save it as `YY-YY.csv` format (e.g., `25-26.csv` for 2025-26 season)
  
- **Other sources**: Any CSV file with columns like:
  - `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG` (full-time goals)
  - `Div` column should be "E0" for Premier League

### Step 2: Add Files to data/ Folder

1. Place your new CSV file(s) in the `data/` folder
2. Name format: `YY-YY.csv` (e.g., `25-26.csv`, `26-27.csv`)
   - The script will automatically extract the season from the filename

### Step 3: Update Combined Dataset

Run the update script:

```bash
python update_data.py
```

This script will:
- ✅ Load all CSV files from `data/` folder
- ✅ Extract season from filename
- ✅ Filter for Premier League matches (Div = "E0")
- ✅ Handle date parsing automatically
- ✅ Remove duplicates (if same match appears in multiple files)
- ✅ Combine everything into `premier_league_combined.csv`
- ✅ Show you the latest matches and date range

### Step 4: Train the Model

Train the model with the updated data (this automatically adds all features):

```bash
python step3.py
```

This will:
- Build all features (rolling stats, Elo, odds, rest days)
- Split data into train/validation/test sets
- Train XGBoost model
- Evaluate performance
- Show predictions on recent games

## File Structure

```
EPL-ML-PREDICTION/
├── data/                          # Put new CSV files here
│   ├── 24-25.csv
│   ├── 25-26.csv
│   └── ... (other seasons)
├── update_data.py                 # Combines all CSV files
├── main.py                        # Adds rolling features
├── step3.py                       # Trains XGBoost model
├── premier_league_combined.csv    # Combined dataset (auto-generated)
└── epl_with_features.csv          # Dataset with features (auto-generated)
```

## Updating Existing Seasons

If you add more games to an existing season file (e.g., update `25-26.csv` with new matches):

1. Replace the old file in `data/` with the new one (or add new rows)
2. Run `python update_data.py` - it will automatically handle duplicates
3. The script keeps the latest version of each match

## Example: Adding 2025-26 Season Games

```bash
# 1. Download latest 25-26.csv from football-data.co.uk
# 2. Place it in data/ folder
# 3. Run update script
python update_data.py

# Output:
# [UPDATE DATA] Combining all season files...
# Found 22 CSV files:
#   Loading 25-26.csv... [OK] 45 matches  (was 30, now has new games)
# ...
# [OK] Saved 5745 matches to premier_league_combined.csv

# 4. Train model (automatically adds all features)
python main.py
```

## Troubleshooting

**Problem:** Script can't extract season from filename
- **Solution:** Make sure filename format is `YY-YY.csv` (e.g., `25-26.csv`)

**Problem:** Dates not parsing correctly
- **Solution:** The script handles multiple date formats automatically. If issues persist, check the CSV file date format.

**Problem:** Too many duplicates
- **Solution:** The script automatically removes duplicates. If you see warnings, check if you have the same match in multiple files.

**Problem:** Missing columns error
- **Solution:** Ensure your CSV has the required columns: `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`

## Data Format Requirements

Your CSV files should have these columns at minimum:

| Column | Description | Example |
|--------|-------------|---------|
| `Date` | Match date | `16/08/2024` or `2024-08-16` |
| `HomeTeam` | Home team name | `Arsenal` |
| `AwayTeam` | Away team name | `Chelsea` |
| `FTHG` | Full-time home goals | `2` |
| `FTAG` | Full-time away goals | `1` |
| `Div` | Division (should be "E0" for Premier League) | `E0` |

Optional but recommended columns:
- `HS`, `AS` (shots)
- `HST`, `AST` (shots on target)
- `B365H`, `B365D`, `B365A` (betting odds)
- `HTHG`, `HTAG` (half-time goals)

## Tips

1. **Regular Updates**: Update weekly/monthly as new games are played
2. **Keep Old Files**: Don't delete old season files - the script uses all of them
3. **Check Latest Matches**: After updating, check the "LATEST MATCHES" output to verify new games were added
4. **Model Performance**: More recent data helps the model learn current team form

