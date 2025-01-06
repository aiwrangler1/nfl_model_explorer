"""Simple data loader for NFL play-by-play data."""

import pandas as pd
import nfl_data_py as nfl

def load_pbp_data(season: int = 2024) -> pd.DataFrame:
    """Load play-by-play data for the specified season."""
    # Load raw data
    df = nfl.import_pbp_data([season])
    
    # Keep only necessary columns
    cols = [
        'game_id', 'play_id', 'posteam', 'defteam', 'week', 'season',
        'drive', 'down', 'yards_gained', 'play_type', 'yardline_100',
        'ydstogo', 'ep', 'epa'
    ]
    
    return df[cols].copy()

def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """Basic data preparation."""
    # Drop plays with missing critical values
    critical_cols = ['posteam', 'defteam', 'drive', 'down', 'yardline_100']
    df = df.dropna(subset=critical_cols)
    
    # Standardize team names
    df.loc[df['posteam'] == 'LA', 'posteam'] = 'LAR'
    df.loc[df['defteam'] == 'LA', 'defteam'] = 'LAR'
    
    return df

def load_and_prepare_data(season: int = 2024) -> pd.DataFrame:
    """Load and prepare data in one step."""
    df = load_pbp_data(season)
    return prepare_data(df) 