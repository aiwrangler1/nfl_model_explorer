# edp_analysis/data_loader.py

"""
NFL Data Loader

This module handles loading and initial preparation of NFL play-by-play data
using nfl_data_py package.
"""

import os
import pandas as pd
import nfl_data_py as nfl
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime

from ..utils.data_validation import validate_required_columns, check_null_values, validate_team_names
from ..utils.data_processing import clean_team_names
from .config import DATA_CONFIG, PROJECT_ROOT, DATA_DIR


def setup_logging(log_level: int = logging.INFO) -> logging.Logger:
    """Set up logging for the data loader with both file and console handlers."""
    logger = logging.getLogger('edp_analysis.data_loader')
    logger.setLevel(log_level)
    
    if not logger.handlers:
        # Create logs directory if it doesn't exist
        log_dir = PROJECT_ROOT / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        # File handler
        log_file = log_dir / f'data_loader_{datetime.now():%Y%m%d}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(logging.Formatter(
            '%(levelname)s - %(message)s'
        ))
        logger.addHandler(console_handler)
    
    return logger


def validate_data_volume(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Validate data volume meets expectations."""
    # Check games per season
    games_per_season = df.groupby('season')['game_id'].nunique()
    for season, count in games_per_season.items():
        if count < DATA_CONFIG.EXPECTED_GAMES_PER_SEASON:
            logger.warning(
                f"Season {season} has fewer games than expected: "
                f"{count} vs {DATA_CONFIG.EXPECTED_GAMES_PER_SEASON}"
            )
    
    # Check plays per game
    plays_per_game = df.groupby('game_id').size()
    low_play_games = plays_per_game[plays_per_game < DATA_CONFIG.MIN_PLAYS_PER_GAME]
    high_play_games = plays_per_game[plays_per_game > DATA_CONFIG.MAX_PLAYS_PER_GAME]
    
    if not low_play_games.empty:
        logger.warning(f"Found {len(low_play_games)} games with suspiciously few plays:")
        for game_id, count in low_play_games.items():
            logger.warning(f"  {game_id}: {count} plays")
    
    if not high_play_games.empty:
        logger.warning(f"Found {len(high_play_games)} games with suspiciously many plays:")
        for game_id, count in high_play_games.items():
            logger.warning(f"  {game_id}: {count} plays")


def validate_numeric_ranges(df: pd.DataFrame, logger: logging.Logger) -> None:
    """Validate numeric columns are within expected ranges."""
    for col, (min_val, max_val) in DATA_CONFIG.NUMERIC_RANGES.items():
        if col in df.columns:
            out_of_range = df[~df[col].between(min_val, max_val)]
            if not out_of_range.empty:
                logger.warning(
                    f"Found {len(out_of_range)} values outside expected range "
                    f"for {col} ({min_val}, {max_val})"
                )
                logger.debug(f"Sample of out-of-range {col} values:")
                for _, row in out_of_range.head().iterrows():
                    logger.debug(f"  {row['game_id']}: {row[col]}")


def get_cache_path(seasons: List[int], weeks: Optional[List[int]] = None) -> Path:
    """Generate cache file path based on seasons and weeks."""
    cache_dir = DATA_DIR / 'cache'
    cache_dir.mkdir(exist_ok=True)
    
    weeks_str = f"_w{'_'.join(map(str, weeks))}" if weeks else ""
    seasons_str = '_'.join(map(str, seasons))
    return cache_dir / f"pbp_s{seasons_str}{weeks_str}.parquet"


def load_pbp_data(seasons: List[int], 
                  weeks: Optional[List[int]] = None,
                  use_cache: bool = True,
                  logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    """
    Load NFL play-by-play data for specified seasons and weeks.
    
    Args:
        seasons: List of seasons to load data for
        weeks: Optional list of weeks to filter by
        use_cache: Whether to use cached data if available
        logger: Optional logger instance
        
    Returns:
        DataFrame containing play-by-play data
    """
    if logger is None:
        logger = setup_logging()
    
    try:
        cache_path = get_cache_path(seasons, weeks)
        
        # Try to load from cache if enabled
        if use_cache and cache_path.exists():
            logger.info(f"Loading cached data from {cache_path}")
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached data: {e}")
        
        # Load fresh data
        logger.info(f"Loading play-by-play data for seasons: {seasons}")
        df = nfl.import_pbp_data(seasons)
        logger.info(f"Loaded {len(df)} plays")
        
        # Filter by weeks if specified
        if weeks:
            logger.info(f"Filtering for weeks: {weeks}")
            df = df[df['week'].isin(weeks)]
            logger.info(f"Filtered to {len(df)} plays")
        
        # Cache the data
        if use_cache:
            logger.info(f"Caching data to {cache_path}")
            df.to_parquet(cache_path)
        
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
        
        # Additional validations
        validate_data_volume(df, logger)
        validate_numeric_ranges(df, logger)
        
        return df
        
    except Exception as e:
        logger.error(f"Error preparing play-by-play data: {str(e)}")
        raise


def load_and_prepare_data(seasons: List[int] = [2024],
                         weeks: Optional[List[int]] = None,
                         use_cache: bool = True,
                         log_level: int = logging.INFO) -> pd.DataFrame:
    """
    Load and prepare NFL play-by-play data.
    
    Args:
        seasons: List of seasons to load data for (default: [2024])
        weeks: Optional list of weeks to filter by
        use_cache: Whether to use cached data if available
        log_level: Logging level to use
        
    Returns:
        Prepared DataFrame ready for EDP analysis
    """
    logger = setup_logging(log_level)
    
    try:
        # Load raw data
        raw_data = load_pbp_data(seasons, weeks, use_cache, logger)
        
        # Prepare data
        prepared_data = prepare_pbp_data(raw_data, logger)
        
        logger.info("Data loading and preparation completed successfully")
        return prepared_data
        
    except Exception as e:
        logger.error(f"Failed to load and prepare data: {str(e)}")
        raise
