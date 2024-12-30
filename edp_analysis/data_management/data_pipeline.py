"""
Data Processing Pipeline for NFL EDP Analysis

This module implements a robust data processing pipeline for NFL play-by-play data,
including validation, transformation, and quality checks at each step.

The pipeline follows these steps:
1. Load raw play-by-play data
2. Validate data completeness and structure
3. Clean and standardize team names
4. Validate numerical ranges and scoring data
5. Calculate drive and game metrics
6. Perform final quality checks

Each step includes detailed logging and error handling to ensure data quality.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import nfl_data_py as nfl
import traceback
import os
import logging
from datetime import datetime
from pathlib import Path

from ..utils.logging_config import setup_logging
from ..utils.data_validation import DataValidator
from ..utils.data_processing import standardize_team_names, calculate_drive_metrics, calculate_game_metrics
from .config import DATA_CONFIG, PROJECT_ROOT


class NFLDataPipeline:
    """
    Data processing pipeline for NFL data.
    
    This pipeline handles the entire process of loading, validating, and transforming
    NFL play-by-play data for EDP analysis. It includes robust error handling and
    detailed logging at each step.
    
    Attributes:
        seasons (List[int]): List of seasons to process
        logger (logging.Logger): Logger for the pipeline
        validator (DataValidator): Data validator instance
        output_dir (Path): Directory for pipeline outputs
    """
    
    def __init__(self, seasons: List[int], log_level: int = logging.INFO) -> None:
        """
        Initialize the NFL data pipeline.
        
        Args:
            seasons: List of seasons to process
            log_level: Logging level (default: logging.INFO)
        """
        self.seasons = seasons
        self.logger = setup_logging(log_level, module_name=__name__)
        self.validator = DataValidator()
        self.output_dir = PROJECT_ROOT / 'pipeline_outputs'
        self.output_dir.mkdir(exist_ok=True)
        
    def load_raw_data(self) -> pd.DataFrame:
        """
        Load raw play-by-play data from nfl_data_py.
        
        Returns:
            DataFrame with raw play-by-play data
            
        Raises:
            ValueError: If play-by-play data validation fails
            Exception: For other loading errors
        """
        try:
            # Load play by play data
            self.logger.info(f"Loading play-by-play data for seasons: {self.seasons}")
            pbp_df = nfl.import_pbp_data(years=self.seasons)
            
            # Validate dataset completeness
            is_valid, validation_results = self.validator.validate_dataset_completeness(
                pbp_df, 
                list(DATA_CONFIG.REQUIRED_COLUMNS), 
                "play_by_play",
                list(DATA_CONFIG.NULLABLE_COLUMNS)
            )
            
            if not is_valid:
                self.logger.error("Play-by-play data validation failed")
                self.logger.error(f"Validation results: {validation_results}")
                
                # Filter out rows with missing required data
                for col in DATA_CONFIG.REQUIRED_COLUMNS:
                    if col not in DATA_CONFIG.NULLABLE_COLUMNS:
                        pbp_df = pbp_df.dropna(subset=[col])
                
                self.logger.info(f"Filtered dataset to {len(pbp_df)} valid rows")
            
            # Validate team names
            team_cols = ['home_team', 'away_team', 'posteam', 'defteam']
            is_valid, invalid_teams = self.validator.validate_team_names(
                pbp_df, 
                team_cols,
                DATA_CONFIG.TEAM_NAME_MAP.keys()
            )
            
            if not is_valid:
                self.logger.warning(f"Found invalid team names: {invalid_teams}")
                # Standardize team names
                pbp_df = standardize_team_names(pbp_df, team_cols, DATA_CONFIG.TEAM_NAME_MAP)
            
            self.logger.info(f"Loaded pbp data: {len(pbp_df)} rows")
            return pbp_df
            
        except Exception as e:
            self.logger.error(f"Error loading raw data: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
            
    def validate_numerical_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate all numerical data in the dataframe.
        
        Args:
            df: Play-by-play dataframe
            
        Returns:
            Tuple of (filtered DataFrame, validation results dictionary)
        """
        try:
            # Validate all numeric ranges from config
            is_valid, validation_results = self.validator.validate_numerical_ranges(
                df, 
                DATA_CONFIG.NUMERIC_RANGES
            )
            
            if not is_valid:
                self.logger.warning("Found invalid numerical values")
                self.logger.warning(f"Validation results: {validation_results}")
                
                # Filter out rows with invalid values
                for col, (min_val, max_val) in DATA_CONFIG.NUMERIC_RANGES.items():
                    if col in df.columns:
                        df = df[
                            df[col].between(min_val, max_val) | 
                            df[col].isnull()
                        ]
                
                self.logger.info(f"Filtered to {len(df)} rows with valid numerical values")
            
            return df, validation_results
                
        except Exception as e:
            self.logger.error(f"Error validating numerical data: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame(), {}
    
    def validate_data_volume(self, df: pd.DataFrame) -> bool:
        """
        Validate data volume meets expectations.
        
        Args:
            df: Play-by-play dataframe
            
        Returns:
            bool indicating if volume checks passed
        """
        try:
            # Check games per season
            games_per_season = df.groupby('season')['game_id'].nunique()
            volume_valid = True
            
            for season, count in games_per_season.items():
                if count < DATA_CONFIG.EXPECTED_GAMES_PER_SEASON:
                    self.logger.warning(
                        f"Season {season} has fewer games than expected: "
                        f"{count} vs {DATA_CONFIG.EXPECTED_GAMES_PER_SEASON}"
                    )
                    volume_valid = False
            
            # Check plays per game
            plays_per_game = df.groupby('game_id').size()
            invalid_games = plays_per_game[
                ~plays_per_game.between(
                    DATA_CONFIG.MIN_PLAYS_PER_GAME,
                    DATA_CONFIG.MAX_PLAYS_PER_GAME
                )
            ]
            
            if not invalid_games.empty:
                self.logger.warning(
                    f"Found {len(invalid_games)} games with suspicious play counts"
                )
                volume_valid = False
            
            return volume_valid
            
        except Exception as e:
            self.logger.error(f"Error validating data volume: {str(e)}")
            return False
    
    def save_pipeline_outputs(self, df: pd.DataFrame, validation_results: Dict) -> None:
        """Save pipeline outputs and validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save processed data
        output_path = self.output_dir / f'processed_pbp_{timestamp}.parquet'
        df.to_parquet(output_path)
        
        # Save validation report
        report_path = self.output_dir / f'validation_report_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write(f"NFL Data Pipeline Validation Report\n")
            f.write(f"Generated: {datetime.now()}\n\n")
            f.write(f"Seasons processed: {self.seasons}\n")
            f.write(f"Final row count: {len(df)}\n\n")
            f.write("Validation Results:\n")
            for key, value in validation_results.items():
                f.write(f"{key}:\n{value}\n\n")
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Run the full NFL data processing pipeline.
        
        Returns:
            Processed play-by-play DataFrame with drive metrics
            
        Note:
            Returns empty DataFrame if any critical step fails.
            Invalid rows are filtered out rather than failing the entire pipeline.
        """
        try:
            # Load raw data
            df = self.load_raw_data()
            if df.empty:
                self.logger.error("Failed to load raw data")
                return pd.DataFrame()
            
            # Validate numerical data
            df, validation_results = self.validate_numerical_data(df)
            if df.empty:
                self.logger.error("Numerical validation failed")
                return pd.DataFrame()
            
            # Validate data volume
            if not self.validate_data_volume(df):
                self.logger.warning("Data volume validation failed")
            
            # Calculate metrics
            drive_metrics = calculate_drive_metrics(df)
            df = pd.merge(df, drive_metrics, on=['game_id', 'drive'], how='left')
            
            # Save outputs
            self.save_pipeline_outputs(df, validation_results)
            
            self.logger.info(f"Pipeline completed successfully with {len(df)} rows")
            return df
                
        except Exception as e:
            self.logger.error(f"Error running pipeline: {str(e)}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()