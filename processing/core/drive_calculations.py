"""
Data Processing Module

This module provides functions for processing NFL play-by-play data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import logging

from ..data_management.config import DATA_CONFIG

logger = logging.getLogger(__name__)


def clean_team_names(df: pd.DataFrame, team_columns: List[str]) -> pd.DataFrame:
    """Standardize team names using the mapping in DATA_CONFIG."""
    df = df.copy()
    
    for col in team_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in DataFrame")
            continue
            
        df[col] = df[col].replace(DATA_CONFIG.TEAM_NAME_MAP)
        
    return df


def calculate_drive_success(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate drive success metrics."""
    df = df.copy()
    
    # Calculate yards needed percentage
    df['yards_needed_pct'] = df['yards_gained'] / df['ydstogo']
    
    # Define success based on down
    df['is_success'] = np.where(
        df['down'].isin([1, 2]),
        df['yards_needed_pct'] >= 0.5,
        df['yards_needed_pct'] >= 1.0
    )
    
    return df 