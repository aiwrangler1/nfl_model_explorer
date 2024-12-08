# edp_analysis/data_loader.py

"""
Data loading and preparation utilities for the NFL EDP analysis.
"""

import pandas as pd
import logging
import nfl_data_py as nfl
from utils.data_validation import DataValidator
from config import TEAM_NAME_MAP, EXPECTED_GAMES_PER_SEASON

def load_and_prepare_data(seasons=None, log_level=logging.INFO):
    """
    Load and perform initial cleaning on NFL play-by-play data.
    
    Args:
        seasons (list): List of seasons to analyze. Defaults to current season if None.
        log_level (int): Logging level to use. Defaults to INFO.
        
    Returns:
        pd.DataFrame: Raw play-by-play data ready for EDP calculation
    """
    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Initialize data validator
    validator = DataValidator()
    
    # Default to current season if no seasons provided
    if seasons is None:
        seasons = [2024]
    
    try:
        # Load data with initial consistency checks
        df_all_seasons = nfl.import_pbp_data(seasons, downcast=True)
        logging.info(f"Initial play-by-play rows: {len(df_all_seasons)}")
        
        # Basic filtering - keep only relevant play types
        df_all_seasons = df_all_seasons[
            df_all_seasons['play_type'].isin(['run', 'pass', 'field_goal'])
        ].copy()
        
        # Add LAS to valid teams for Las Vegas Raiders
        validator.valid_teams.add('LAS')
        
        # Validate dataset completeness
        required_columns = [
            'game_id', 'play_id', 'posteam', 'defteam', 'week', 'season',
            'drive', 'down', 'yards_gained', 'play_type', 'yardline_100', 'ydstogo',
            'touchdown', 'field_goal_result', 'extra_point_result', 'two_point_conv_result',
            'safety', 'field_goal_attempt'
        ]
        is_valid, validation_results = validator.validate_dataset_completeness(
            df_all_seasons,
            required_columns,
            'play_by_play',
            null_allowed_columns=['down', 'yards_gained']
        )
        
        if not is_valid:
            logging.warning(f"Data validation issues found: {validation_results}")
        
        # Validate team names
        team_cols = ['posteam', 'defteam', 'home_team', 'away_team']
        team_validation_results = validator.validate_team_names(df_all_seasons, team_cols)
        if not team_validation_results['is_valid']:
            logging.warning(f"Team name validation issues: {team_validation_results['invalid_teams']}")
        
        # Load and validate schedule data
        df_schedules = nfl.import_schedules(seasons)
        expected_games = len(seasons) * EXPECTED_GAMES_PER_SEASON
        actual_games = len(df_schedules)
        if actual_games < expected_games * 0.9:  # Allow for 10% missing data
            logging.warning(
                f"Schedule data may be incomplete. Expected ~{expected_games}, got {actual_games}"
            )
        
        logging.info("Data preparation completed successfully")
        return df_all_seasons
        
    except Exception as e:
        logging.error(f"Error in data preparation: {str(e)}")
        raise
