# edp_analysis/data_loader.py

import nfl_data_py as nfl
import pandas as pd
import logging

def load_and_prepare_data():
    """
    Load and perform initial cleaning on NFL play-by-play data for the seasons 2013 to 2023,
    including playoffs.
    """
    # Define the season range from 2013 to 2023
    # seasons = list(range(2013, 2024))
    seasons = [2024]


    try:
        # Download play-by-play data for the specified seasons
        df_all_seasons = nfl.import_pbp_data(seasons)
        
        # Print all available columns for debugging
        print("\nAvailable columns in raw data:")
        print(sorted(df_all_seasons.columns.tolist()))
        
        print(f"\nData loaded successfully. Total rows: {len(df_all_seasons)}")

        # Before processing play types, add a check for None values
        df_all_seasons['play_type'] = df_all_seasons['play_type'].fillna('unknown')  # Replace None with 'unknown'
        
        # When displaying unique play types, add error handling
        try:
            print("\nUnique play types in the data:")
            print(sorted(df_all_seasons['play_type'].unique()))  # Sort the play types for better readability
        except Exception as e:
            logging.warning(f"Warning when processing play types: {e}")
            # Continue execution instead of failing
            pass

        # Columns needed for EDP calculations and other analysis
        required_columns = [
            'play_type', 'posteam', 'defteam', 'yards_gained', 'down', 'ydstogo', 
            'yardline_100', 'epa', 'game_id', 'drive', 'play_id', 'home_team', 
            'away_team', 'total_home_score', 'total_away_score', 'season', 'game_date'
        ]

        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df_all_seasons.columns]
        if missing_columns:
            print(f"\nMissing required columns: {missing_columns}")
            return None

        # Print sample of first few rows for debugging
        print("\nSample of first few rows:")
        print(df_all_seasons[required_columns].head())

        # Retain only necessary columns but DON'T filter play types yet
        df_all_seasons = df_all_seasons[required_columns].dropna()

        # Calculate score differential
        df_all_seasons['score_differential'] = df_all_seasons.apply(
            lambda row: row['total_home_score'] - row['total_away_score'] 
            if row['posteam'] == row['home_team'] 
            else row['total_away_score'] - row['total_home_score'], 
            axis=1
        )

        # Add field goal result column
        df_all_seasons['field_goal_result'] = df_all_seasons.apply(
            lambda row: 'made' if row['play_type'] == 'field_goal' and row['score_differential'] != 0 else 'missed',
            axis=1
        )
        
        # Add field goal attempt column
        df_all_seasons['field_goal_attempt'] = (df_all_seasons['play_type'] == 'field_goal').astype(int)
        
        # Add game seconds remaining based on drive clock
        if 'drive_game_clock_start' in df_all_seasons.columns:
            df_all_seasons['game_seconds_remaining'] = df_all_seasons['drive_game_clock_start']
        else:
            # Default to 3600 seconds (60 minutes) at start of game
            df_all_seasons['game_seconds_remaining'] = 3600

        print(f"Data prepared successfully. Total rows after cleaning: {len(df_all_seasons)}")
        print("\nPlay types in final dataset:")
        print(sorted(df_all_seasons['play_type'].unique().tolist()))

        return df_all_seasons

    except Exception as e:
        print(f"\nError loading data: {str(e)}")
        return None

if __name__ == "__main__":
    data = load_and_prepare_data()
    if data is None:
        print("Failed to load data properly")
