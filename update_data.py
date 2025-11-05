"""
Update Premier League dataset with newer CSV files.
This script:
1. Loads all CSV files from the data/ folder
2. Combines them into premier_league_combined.csv
3. Handles duplicates and date parsing
4. Updates the combined dataset
"""

import pandas as pd
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Paths
DATA_DIR = Path("data")
COMBINED_CSV = Path("premier_league_combined.csv")

def extract_season_from_filename(filename: str) -> str:
    """Extract season from filename like '24-25.csv' -> '2024-25' or '5-6.csv' -> '2005-06'"""
    match = re.search(r'(\d{1,2})-(\d{1,2})', filename)
    if match:
        start_year = int(match.group(1))
        # Convert 1 or 2-digit year to 4-digit
        if start_year < 50:
            # 1-digit or 2-digit years < 50 are 2000s
            start_year = 2000 + start_year
        elif start_year < 100:
            # 2-digit years 50-99 are 1900s (shouldn't happen for EPL, but handle it)
            start_year = 1900 + start_year
        # If already 4-digit, use as-is
        
        end_year = start_year + 1
        return f"{start_year}-{str(end_year)[2:]}"
    return None

def parse_date_robust(s: pd.Series) -> pd.Series:
    """Robust date parsing for various formats"""
    s = s.astype(str)
    # ISO format YYYY-MM-DD
    mask_iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$")
    out = pd.to_datetime(s.where(mask_iso), format="%Y-%m-%d", errors="coerce")
    # European format DD/MM/YYYY
    out2 = pd.to_datetime(s.where(~mask_iso), dayfirst=True, errors="coerce")
    out = out.fillna(out2)
    return out

def load_and_prepare_csv(filepath: Path, season: str) -> pd.DataFrame:
    """Load a CSV file and prepare it for merging"""
    try:
        print(f"  Loading {filepath.name}...", end=" ")
        df = pd.read_csv(filepath, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Add Season column if not present
        if "Season" not in df.columns:
            df["Season"] = season
        
        # Ensure Date is datetime
        if "Date" in df.columns:
            df["Date"] = parse_date_robust(df["Date"])
        
        # Filter only E0 (Premier League) matches if Div column exists
        if "Div" in df.columns:
            df = df[df["Div"] == "E0"].copy()
        
        print(f"[OK] {len(df)} matches")
        return df
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return pd.DataFrame()

def combine_all_seasons():
    """Combine all CSV files from data/ folder into one dataset"""
    print("[UPDATE DATA] Combining all season files...\n")
    
    if not DATA_DIR.exists():
        print(f"Error: {DATA_DIR} directory not found!")
        return
    
    # Find all CSV files
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {DATA_DIR}")
        return
    
    print(f"Found {len(csv_files)} CSV files:\n")
    
    all_dataframes = []
    seen_seasons = set()
    
    for csv_file in csv_files:
        # Extract season from filename
        season = extract_season_from_filename(csv_file.name)
        
        if not season:
            print(f"  Warning: Could not extract season from {csv_file.name}, skipping...")
            continue
        
        # Load and prepare
        df = load_and_prepare_csv(csv_file, season)
        
        if not df.empty:
            # Store season for deduplication
            df["_source_file"] = csv_file.name
            all_dataframes.append(df)
            seen_seasons.add(season)
    
    if not all_dataframes:
        print("\nNo valid data loaded!")
        return
    
    print(f"\n[COMBINING] Merging {len(all_dataframes)} files...")
    combined = pd.concat(all_dataframes, ignore_index=True, sort=False)
    
    print(f"Total rows before deduplication: {len(combined)}")
    
    # Remove duplicates based on Season, Date, HomeTeam, AwayTeam
    if all(col in combined.columns for col in ["Season", "Date", "HomeTeam", "AwayTeam"]):
        # Sort by source file (prefer later files) and date
        combined = combined.sort_values(["_source_file", "Date"], ascending=[False, True])
        
        # Remove duplicates, keeping first (which will be from later files)
        combined = combined.drop_duplicates(
            subset=["Season", "Date", "HomeTeam", "AwayTeam"],
            keep="first"
        )
        
        print(f"Total rows after deduplication: {len(combined)}")
    
    # Drop temporary column
    if "_source_file" in combined.columns:
        combined = combined.drop(columns=["_source_file"])
    
    # Sort by season and date
    if "Season" in combined.columns and "Date" in combined.columns:
        combined = combined.sort_values(["Season", "Date"]).reset_index(drop=True)
    
    # Save
    print(f"\n[SAVING] Writing to {COMBINED_CSV}...")
    combined.to_csv(COMBINED_CSV, index=False)
    
    print(f"[OK] Saved {len(combined)} matches to {COMBINED_CSV}")
    print(f"\nSeasons included: {sorted(seen_seasons)}")
    print(f"Date range: {combined['Date'].min()} to {combined['Date'].max()}")
    
    # Show latest matches
    print("\n[LATEST MATCHES]")
    latest = combined.tail(10)[["Season", "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]]
    print(latest.to_string(index=False))
    
    print("\n[DONE] Dataset updated successfully!")
    print("\nNext step:")
    print("Run: python main.py  (this will add all features and train the model)")

if __name__ == "__main__":
    combine_all_seasons()

