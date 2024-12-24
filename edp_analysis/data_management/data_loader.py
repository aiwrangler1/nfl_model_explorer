# edp_analysis/data_loader.py

"""
NFL Data Loader

This module handles loading and initial preparation of NFL play-by-play data
using nfl_data_py package.
"""

import pandas as pd
import nfl_data_py as nfl
import logging
from typing import List, Optional

from ..utils.data_validation import validate_required_columns, check_null_values, validate_team_names
from ..utils.data_processing import clean_team_names
from .config import DATA_CONFIG


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Set up logging for the data loader."""
    logger = logging.getLogger('edp_analysis.data_loader')
    logger.setLevel(log_level)
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_pbp_data(seasons: List[int], 
                  weeks: Optional[List[int]] = None,
                  logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load NFL play-by-play data for specified seasons and weeks.
    
    Args:
        seasons: List of seasons to load data for
        weeks: Optional list of weeks to filter by
        logger: Optional logger instance
        
    Returns:
        DataFrame containing play-by-play data
    """
    if logger is None:
        logger = setup_logging()
    
    try:
        # Load data using nfl_data_py
        logger.info(f"Loading play-by-play data for seasons: {seasons}")
        df = nfl.import_pbp_data(seasons)
        logger.info(f"Loaded {len(df)} plays")
        
        # Filter by weeks if specified
        if weeks:
            logger.info(f"Filtering for weeks: {weeks}")
            df = df[df['week'].isin(weeks)]
            logger.info(f"Filtered to {len(df)} plays")
        
        return df
        
    except Exception as e:
        logger.error(f"Error loading play-by-play data: {str(e)}")
        raise


def prepare_pbp_data(df: pd.DataFrame,
                    logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Prepare play-by-play data for EDP analysis.
    
    Args:
        df: Raw play-by-play DataFrame
        logger: Optional logger instance
        
    Returns:
        Prepared DataFrame
    """
    if logger is None:
        logger = setup_logging()
    
    try:
        # Validate required columns
        missing_cols = validate_required_columns(df, DATA_CONFIG.REQUIRED_COLUMNS)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Clean team names
        logger.info("Standardizing team names")
        df = clean_team_names(df, ['posteam', 'defteam'])
        
        # Validate team names
        invalid_teams = validate_team_names(df, 'posteam')
        if invalid_teams:
            logger.warning(f"Found invalid offensive team names: {invalid_teams}")
        
        invalid_teams = validate_team_names(df, 'defteam')
        if invalid_teams:
            logger.warning(f"Found invalid defensive team names: {invalid_teams}")
        
        # Filter to valid play types
        logger.info("Filtering to valid play types")
        df = df[df['play_type'].isin(DATA_CONFIG.VALID_PLAY_TYPES)]
        logger.info(f"Retained {len(df)} valid plays")
        
        # Check for null values in critical columns
        null_counts = check_null_values(
            df,
            [col for col in DATA_CONFIG.REQUIRED_COLUMNS 
             if col not in DATA_CONFIG.NULLABLE_COLUMNS]
        )
        if null_counts:
            logger.warning("Found null values in critical columns:")
            for col, count in null_counts.items():
                logger.warning(f"  {col}: {count} nulls")
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing play-by-play data: {str(e)}")
        raise


def load_and_prepare_data(seasons: List[int] = [2024],
                         weeks: Optional[List[int]] = None,
                         log_level: int = logging.INFO) -> pd.DataFrame:
    """
    Load and prepare NFL play-by-play data.
    
    Args:
        seasons: List of seasons to load data for (default: [2024])
        weeks: Optional list of weeks to filter by
        log_level: Logging level to use
        
    Returns:
        Prepared DataFrame ready for EDP analysis
    """
    logger = setup_logging(log_level)
    
    try:
        # Load raw data
        raw_data = load_pbp_data(seasons, weeks, logger)
        
        # Prepare data
        prepared_data = prepare_pbp_data(raw_data, logger)
        
        logger.info("Data loading and preparation completed successfully")
        return prepared_data
        
    except Exception as e:
        logger.error(f"Failed to load and prepare data: {str(e)}")
        raise
