"""
Data Validation Module

This module provides functions for validating NFL play-by-play data.
"""

import pandas as pd
from typing import Set, Dict, List
import logging

from ..data_management.config import DATA_CONFIG

logger = logging.getLogger(__name__)


def validate_required_columns(df: pd.DataFrame, required_columns: Set[str]) -> Set[str]:
    """Check if all required columns are present in the DataFrame."""
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        logger.warning(f"Missing required columns: {missing_cols}")
    return missing_cols


def check_null_values(df: pd.DataFrame, critical_columns: List[str]) -> Dict[str, int]:
    """Check for null values in critical columns."""
    null_counts = df[critical_columns].isnull().sum()
    null_cols = {col: count for col, count in null_counts.items() if count > 0}
    
    if null_cols:
        logger.warning("Found null values in critical columns:")
        for col, count in null_cols.items():
            logger.warning(f"  {col}: {count} nulls")
    
    return null_cols


def validate_team_names(df: pd.DataFrame, team_col: str) -> Set[str]:
    """Check for invalid team names in a column."""
    valid_teams = set(DATA_CONFIG.TEAM_NAME_MAP.values())
    found_teams = set(df[team_col].unique())
    invalid_teams = found_teams - valid_teams - set(DATA_CONFIG.TEAM_NAME_MAP.keys())
    
    if invalid_teams:
        logger.warning(f"Found invalid team names in {team_col}: {invalid_teams}")
    
    return invalid_teams 