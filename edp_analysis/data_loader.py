# edp_analysis/data_loader.py

"""
Data loading and preparation utilities for the NFL EDP analysis.
"""

import pandas as pd
import logging
import nfl_data_py as nfl
from utils.data_validation import validate_dataset_completeness, validate_team_names, validate_numerical_ranges
from utils.data_processing import standardize_team_names, calculate_drive_metrics, calculate_game_metrics
from config import TEAM_NAME_MAP, EXPECTED_GAMES_PER_SEASON, MIN_ROSTER_SIZE, KEY_POSITIONS, OFFENSIVE_POSITIONS

def load_and_prepare_data(seasons=None, log_level=logging.INFO):
    """
    Load and perform initial cleaning on NFL play-by-play data.
    
    Args:
        seasons (list): List of seasons to analyze. Defaults to current season if None.
        log_level (int): Logging level to use. Defaults to INFO.
    """
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Default to current season if no seasons provided
    if seasons is None:
        seasons = [2024]
    
    try:
        # Load data with initial consistency checks
        df_all_seasons = nfl.import_pbp_data(seasons, downcast=True)
        logging.info(f"Initial play-by-play rows: {len(df_all_seasons)}")
        
        # Standardize team names in play-by-play data
        team_cols = ['posteam', 'defteam', 'home_team', 'away_team']
        df_all_seasons = standardize_team_names(df_all_seasons, team_cols, TEAM_NAME_MAP)
        
        # Load and standardize schedule data
        df_schedules = nfl.import_schedules(seasons)
        df_schedules = standardize_team_names(df_schedules, ['home_team', 'away_team'], TEAM_NAME_MAP)
        
        # Verify schedule data completeness
        expected_games = len(seasons) * EXPECTED_GAMES_PER_SEASON
        actual_games = len(df_schedules)
        if actual_games < expected_games * 0.9:  # Allow for 10% missing data
            logging.warning(
                f"Schedule data may be incomplete. Expected ~{expected_games}, got {actual_games}"
            )
        
        # Load and verify roster data
        df_rosters = nfl.import_seasonal_rosters(seasons)
        df_rosters = standardize_team_names(df_rosters, 'team', TEAM_NAME_MAP)
        
        # Verify roster completeness
        players_per_team = df_rosters.groupby(['season', 'team']).size()
        low_roster_teams = players_per_team[players_per_team < MIN_ROSTER_SIZE].reset_index()
        if not low_roster_teams.empty:
            logging.warning(f"Teams with suspiciously small rosters:\n{low_roster_teams}")
        
        # Load and standardize injury data
        df_injuries = nfl.import_injuries(seasons)
        df_injuries = standardize_team_names(df_injuries, 'team', TEAM_NAME_MAP)
        
        # Verify temporal alignment of injuries
        df_injuries['injury_week'] = df_injuries['week'].astype(int)
        invalid_weeks = df_injuries[
            (df_injuries['injury_week'] < 1) | 
            (df_injuries['injury_week'] > 18)
        ]
        if not invalid_weeks.empty:
            logging.warning(
                f"Found {len(invalid_weeks)} injuries with invalid week numbers"
            )
        
        # Merge schedule data with consistency check
        pre_merge_rows = len(df_all_seasons)
        df_all_seasons = pd.merge(
            df_all_seasons,
            df_schedules[['game_id', 'week', 'season', 'gameday', 'gametime', 'home_team', 'away_team']],
            on=['game_id', 'season', 'home_team', 'away_team'],
            how='left',
            validate='m:1'  # many-to-one relationship
        )
        
        # Check for data loss after merge
        post_merge_rows = len(df_all_seasons)
        if post_merge_rows != pre_merge_rows:
            logging.warning(
                f"Row count changed after schedule merge: {pre_merge_rows} -> {post_merge_rows}"
            )
        
        # Check for null values in key columns after merge
        key_columns = ['game_id', 'week', 'gameday', 'home_team', 'away_team']
        null_counts = df_all_seasons[key_columns].isnull().sum()
        if null_counts.any():
            logging.warning(
                f"Null values found after schedule merge:\n{null_counts[null_counts > 0]}"
            )
        
        # Merge injury data with temporal alignment
        injury_summary = df_injuries.groupby(['season', 'week', 'team']).agg({
            'player_id': 'count',  # total injuries
            'status': lambda x: (x.str.lower().isin(['out', 'ir'])).sum()  # serious injuries
        }).reset_index()
        
        injury_summary.columns = ['season', 'week', 'team', 'total_injuries', 'serious_injuries']
        
        # Ensure injuries from previous weeks are reflected in current week
        injury_summary = injury_summary.sort_values(['season', 'team', 'week'])
        injury_summary['cumulative_injuries'] = injury_summary.groupby(['season', 'team'])['serious_injuries'].cumsum()
        
        # Merge injury data with main dataset
        df_all_seasons = pd.merge(
            df_all_seasons,
            injury_summary,
            left_on=['season', 'week', 'posteam'],
            right_on=['season', 'week', 'team'],
            how='left'
        )
        
        # Fill missing injury data with 0
        injury_columns = ['total_injuries', 'serious_injuries', 'cumulative_injuries']
        df_all_seasons[injury_columns] = df_all_seasons[injury_columns].fillna(0)
        
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
            'away_team', 'total_home_score', 'total_away_score', 'season', 'game_date',
            'week', 'gameday', 'gametime', 'injury_count', 'is_home_team'
        ]
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df_all_seasons.columns]
        if missing_columns:
            print(f"\nMissing required columns: {missing_columns}")
            return None
        
        # Print sample of first few rows for debugging
        print("\nSample of first few rows:")
        print(df_all_seasons[required_columns].head())
        
        # More selective NA handling - specify which columns must have values
        critical_columns = ['play_type', 'posteam', 'down', 'ydstogo', 'yardline_100']
        if df_all_seasons[critical_columns].isnull().any().any():
            logging.warning("Critical columns contain null values")
            df_all_seasons = df_all_seasons.dropna(subset=critical_columns)
        
        # Retain only necessary columns with more flexible NA handling
        df_all_seasons = df_all_seasons[required_columns]
        
        # Calculate score differential
        df_all_seasons['score_differential'] = df_all_seasons.apply(
            lambda row: row['total_home_score'] - row['total_away_score'] 
            if row['posteam'] == row['home_team'] 
            else row['total_away_score'] - row['total_home_score'], 
            axis=1
        )
        
        # Improved field goal result calculation using kick_result if available
        if 'kick_result' in df_all_seasons.columns:
            df_all_seasons['field_goal_result'] = df_all_seasons.apply(
                lambda row: row['kick_result'] if row['play_type'] == 'field_goal' 
                else 'not_applicable',
                axis=1
            )
        else:
            logging.warning(
                "kick_result column not available, using score differential for field goal results"
            )
            df_all_seasons['field_goal_result'] = df_all_seasons.apply(
                lambda row: 'made' if row['play_type'] == 'field_goal' and row['score_differential'] != 0 
                else 'missed',
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
        
        # Create more detailed injury metrics
        def categorize_injury_status(status):
            if pd.isna(status):
                return 'unknown'
            status = status.lower()
            if status in ['out', 'ir', 'injured reserve']:
                return 'out'
            elif status in ['doubtful', 'questionable']:
                return 'limited'
            elif status == 'probable':
                return 'probable'
            return 'active'
        
        # Add position information to injury data
        df_injuries['status_category'] = df_injuries['status'].apply(categorize_injury_status)
        
        # Create position-specific injury summaries
        position_injuries = df_injuries.merge(
            df_rosters[['player_id', 'position', 'season']],
            on=['player_id', 'season'],
            how='left'
        )
        
        # Summarize injuries by team, week, and position
        injury_metrics = []
        
        for pos in KEY_POSITIONS:
            pos_injuries = position_injuries[position_injuries['position'] == pos]
            
            # Count serious injuries (out/IR) by position
            serious_injuries = pos_injuries[pos_injuries['status_category'] == 'out']
            pos_summary = serious_injuries.groupby(['season', 'week', 'team']).size().reset_index()
            pos_summary = pos_summary.rename(columns={0: f'{pos.lower()}_injuries'})
            injury_metrics.append(pos_summary)
        
        # Combine all position-specific injury metrics
        combined_injuries = injury_metrics[0]
        for metric in injury_metrics[1:]:
            combined_injuries = pd.merge(
                combined_injuries,
                metric,
                on=['season', 'week', 'team'],
                how='outer'
            )
        
        # Fill NaN values with 0
        combined_injuries = combined_injuries.fillna(0)
        
        # Merge injury metrics with main dataset
        df_all_seasons = pd.merge(
            df_all_seasons,
            combined_injuries,
            left_on=['season', 'week', 'posteam'],
            right_on=['season', 'week', 'team'],
            how='left'
        )
        
        # Fill NaN values for injury columns
        injury_columns = [f'{pos.lower()}_injuries' for pos in KEY_POSITIONS]
        df_all_seasons[injury_columns] = df_all_seasons[injury_columns].fillna(0)
        
        # Add QB injury flag
        df_all_seasons['starting_qb_injured'] = df_all_seasons['qb_injuries'] > 0
        
        # Add total offensive injuries metric
        df_all_seasons['total_offensive_injuries'] = df_all_seasons[[f'{pos}_injuries' for pos in OFFENSIVE_POSITIONS]].sum(axis=1)
        
        # Update required columns list
        required_columns.extend([
            'qb_injuries', 'wr_injuries', 'rb_injuries', 'te_injuries', 'ol_injuries',
            'starting_qb_injured', 'total_offensive_injuries'
        ])
        
        # Calculate game-level metrics
        game_metrics = calculate_game_metrics(df_all_seasons)
        
        # Merge game metrics back to play-by-play data
        df_all_seasons = pd.merge(
            df_all_seasons,
            game_metrics,
            on=['game_id', 'posteam'],
            how='left'
        )
        
        # Add new columns to required columns list
        required_columns.extend([
            'game_avg_epa', 'game_total_epa', 'game_epa_volatility',
            'game_avg_yards', 'game_total_yards', 'game_yards_volatility',
            'game_fg_attempts', 'game_fg_made', 'game_fg_success_rate',
            'game_total_plays', 'final_score_differential',
            'yards_per_play', 'epa_per_play'
        ])
        
        # Final consistency check
        final_null_check = df_all_seasons[required_columns].isnull().sum()
        if final_null_check.any():
            logging.warning("Null values in final dataset:")
            logging.warning(final_null_check[final_null_check > 0])
        
        print(f"Data prepared successfully. Total rows after cleaning: {len(df_all_seasons)}")
        print("\nPlay types in final dataset:")
        print(sorted(df_all_seasons['play_type'].unique().tolist()))
        
        return df_all_seasons
    
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        return None
        
if __name__ == "__main__":
    # Example usage with specific seasons
    data = load_and_prepare_data(seasons=[2023, 2024], log_level=logging.DEBUG)
    if data is None:
        logging.error("Failed to load data properly")
