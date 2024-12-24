"""
Enhanced Earned Drive Points (EDP) Calculator V2

This module implements an improved version of the EDP calculator with better modularity
and separation of concerns.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict


@dataclass
class EDPWeights:
    """Weights for EDP component calculations."""
    # Team weights for total EDP
    OFFENSE_WEIGHT: float = 0.6
    DEFENSE_WEIGHT: float = 0.4


class EDPCalculator:
    """Enhanced EDP Calculator with improved modularity and separation of concerns."""
    
    def __init__(self, weights: EDPWeights = None):
        """Initialize the EDP calculator with optional custom weights."""
        self.weights = weights or EDPWeights()

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare play-by-play data for EDP calculations."""
        df = df.copy()
        
        # Create unique drive identifier
        df['drive_num'] = df['game_id'] + '_' + df['drive'].astype(str)
        
        # Calculate initial EPA-based drive points
        df['earned_drive_points'] = df.groupby('drive_num')['epa'].transform('sum')
        
        return df

    def calculate_drive_quality(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drive quality metrics."""
        # Preprocess data
        df = self.preprocess_data(df)
        
        # Calculate drive-level metrics
        drive_metrics = df.groupby('drive_num').agg({
            'earned_drive_points': 'first',
            'posteam': 'first',
            'defteam': 'first',
            'game_id': 'first',
            'week': 'first',
            'season': 'first'
        }).reset_index()
        
        return drive_metrics

    def calculate_team_metrics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate team-level offensive and defensive EDP metrics."""
        # Calculate drive quality first
        drive_metrics = self.calculate_drive_quality(df)
        
        # Calculate offensive metrics
        offensive_metrics = drive_metrics.groupby(['posteam', 'game_id']).agg({
            'earned_drive_points': 'sum',
            'drive_num': 'count',
            'week': 'first',
            'season': 'first'
        }).reset_index()
        
        offensive_metrics = offensive_metrics.rename(columns={
            'posteam': 'team',
            'drive_num': 'drive_count_off',
            'earned_drive_points': 'earned_drive_points_off'
        })
        
        # Calculate defensive metrics
        defensive_metrics = drive_metrics.groupby(['defteam', 'game_id']).agg({
            'earned_drive_points': 'sum',
            'drive_num': 'count'
        }).reset_index()
        
        defensive_metrics = defensive_metrics.rename(columns={
            'defteam': 'team',
            'drive_num': 'drive_count_def',
            'earned_drive_points': 'earned_drive_points_def'
        })
        
        return offensive_metrics, defensive_metrics