"""
Data processing pipeline for the NFL EDP analysis.
"""

from typing import List, Dict, Optional
import pandas as pd
import nfl_data_py as nfl
import traceback
import os
import logging
from datetime import datetime
from utils.logging_config import setup_logging
from utils.data_validation import DataValidator
from utils.data_processing import standardize_team_names, calculate_drive_metrics, calculate_game_metrics

class NFLDataPipeline:
    """
    Data processing pipeline for NFL data.
    
    Attributes:
        seasons (List[int]): List of seasons to process.
        logger (logging.Logger): Logger for the pipeline.
        validator (DataValidator): Data validator for the pipeline.
        team_name_map (Dict[str, str]): Mapping of non-standard to standard team names.
    """
    
    def __init__(self, seasons: List[int], log_level: int = logging.INFO) -> None:
        """
        Initialize the NFL data pipeline.
        
        Args:
            seasons (List[int]): List of seasons to process.
            log_level (int): Logging level for the pipeline. Defaults to logging.INFO.
        """
        self.seasons = seasons
        self.logger = setup_logging(log_level, module_name=__name__)
        self.validator = DataValidator()
        self.team_name_map = {
            'JAX': 'JAC', 'LA': 'LAR', 'LV': 'LAS', 'SD': 'LAC',
            'STL': 'LAR', 'OAK': 'LAS', 'JAG': 'JAC'
        }
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw play-by-play data from nfl_data_py.
        
        Returns:
            pd.DataFrame: Raw play-by-play data with standardized team names.
            
        Raises:
            ValueError: If play-by-play data validation fails.
            Exception: If there's an error loading the data.
        """
        try:
            # Load play by play data
            pbp_df = nfl.import_pbp_data(years=self.seasons)
            
            # Define required columns and columns where nulls are allowed
            required_cols = [
                'play_id', 'game_id', 'week', 'season',
                'home_team', 'away_team', 'posteam', 'defteam',
                'game_date', 'quarter_seconds_remaining', 
                'half_seconds_remaining', 'game_seconds_remaining',
                'desc', 'play_type', 'yards_gained'
            ]
            
            null_allowed_cols = [
                'timeout', 'challenge', 'penalty', 'penalty_yards',
                'fumble', 'fumble_recovery', 'safety', 'touchdown'
            ]
            
            # Validate dataset completeness
            is_valid, validation_results = self.validator.validate_dataset_completeness(
                pbp_df, required_cols, "play_by_play", null_allowed_cols
            )
            
            if not is_valid:
                self.logger.error("Play-by-play data validation failed")
                self.logger.error(f"Validation results: {validation_results}")
                
                # Filter out rows with missing required data
                for col in required_cols:
                    if col not in null_allowed_cols:
                        pbp_df = pbp_df.dropna(subset=[col])
                
                self.logger.info(f"Filtered dataset to {len(pbp_df)} valid rows")
            
            # Validate team names
            team_cols = ['home_team', 'away_team', 'posteam', 'defteam']
            is_valid, invalid_teams = self.validator.validate_team_names(pbp_df, team_cols)
            
            if not is_valid:
                self.logger.warning(f"Found invalid team names: {invalid_teams}")
                # Standardize team names
                pbp_df = standardize_team_names(pbp_df, team_cols, self.team_name_map)
            
            self.logger.info(f"Loaded pbp data: {len(pbp_df)} rows")
            return pbp_df
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def validate_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate scoring data in the play-by-play dataframe.
        
        Args:
            df (pd.DataFrame): Play-by-play dataframe.
            
        Returns:
            pd.DataFrame: Validated play-by-play dataframe.
            
        Note:
            Will filter out rows with invalid score differentials.
            Warns if invalid score differentials are found.
        """
        try:
            # Validate score differential ranges
            is_valid, validation_results = self.validator.validate_numerical_ranges(
                df, 
                {
                    'score_differential': (-100, 100),
                    'yards_gained': (-50, 100),
                    'play_clock': (0, 40)
                }
            )
            
            if not is_valid:
                self.logger.warning("Found invalid numerical values")
                self.logger.warning(f"Validation results: {validation_results}")
                
                # Filter out rows with invalid score differentials
                df = df[
                    (df['score_differential'].between(-100, 100)) &
                    (df['yards_gained'].between(-50, 100)) &
                    ((df['play_clock'].between(0, 40)) | df['play_clock'].isnull())
                ]
                
                self.logger.info(f"Filtered to {len(df)} rows with valid numerical values")
                
            return df
                
        except Exception as e:
            self.logger.error(f"Error validating scoring: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
            
    def run_pipeline(self) -> pd.DataFrame:
        """
        Run the full NFL data processing pipeline.
        
        Returns:
            pd.DataFrame: Processed play-by-play dataframe with drive metrics.
            
        Note:
            Returns empty DataFrame if any step fails.
            Steps include:
            1. Loading raw data
            2. Validating scoring
            3. Calculating drive metrics
            
            Invalid rows are filtered out rather than failing the entire pipeline.
        """
        try:
            # Load raw data
            df = self.load_raw_data()
            if df.empty:
                self.logger.error("Failed to load raw data")
                return pd.DataFrame()
            
            # Validate scoring
            df = self.validate_scoring(df)
            if df.empty:
                self.logger.error("Scoring validation failed")
                return pd.DataFrame()
            
            # Calculate drive metrics
            drive_metrics = calculate_drive_metrics(df)
            df = pd.merge(df, drive_metrics, on=['game_id', 'drive'], how='left')
            
            self.logger.info(f"Pipeline completed successfully with {len(df)} rows")
            return df
                
        except Exception as e:
            self.logger.error(f"Error running pipeline: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()