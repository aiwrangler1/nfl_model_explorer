"""
Opponent strength adjustment calculations for EDP metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .config import OPPONENT_STRENGTH_SETTINGS

class OpponentStrengthAdjuster:
    def __init__(self, settings: Optional[Dict] = None):
        """
        Initialize the opponent strength adjuster.
        
        Args:
            settings: Optional dictionary of settings to override defaults
        """
        self.settings = settings or OPPONENT_STRENGTH_SETTINGS
        
    def calculate_opponent_strength_metrics(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        window: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate rolling opponent strength metrics.
        
        Args:
            df: DataFrame with team metrics
            metrics: List of metrics to calculate opponent strength for
            window: Rolling window size for calculations
            
        Returns:
            DataFrame with opponent strength metrics
        """
        metrics = metrics or self.settings['metrics']
        window = window or self.settings['window']
        
        opponent_strength = pd.DataFrame()
        
        for metric in metrics:
            # Calculate rolling average of the metric
            rolling_metric = df.groupby('team')[metric].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            opponent_strength[f'opponent_{metric}'] = rolling_metric
            
        return opponent_strength
    
    def calculate_matchup_adjustments(
        self,
        team_metrics: pd.DataFrame,
        opponent_metrics: pd.DataFrame,
        weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Calculate adjustments based on opponent strength.
        
        Args:
            team_metrics: DataFrame with team metrics
            opponent_metrics: DataFrame with opponent strength metrics
            weights: Dictionary of weights for different components
            
        Returns:
            DataFrame with matchup adjustments
        """
        weights = weights or self.settings['weights']
        adjustments = pd.DataFrame()
        
        for metric, weight in weights.items():
            team_value = team_metrics[metric]
            opponent_value = opponent_metrics[f'opponent_{metric}']
            
            # Calculate adjustment based on difference from average
            league_avg = team_value.mean()
            relative_strength = (opponent_value - league_avg) / opponent_value.std()
            
            adjustments[f'{metric}_adjustment'] = relative_strength * weight
            
        return adjustments
    
    def apply_opponent_adjustments(
        self,
        df: pd.DataFrame,
        metric: str,
        opponent_strength: pd.DataFrame,
        matchup_adjustments: pd.DataFrame
    ) -> pd.Series:
        """
        Apply opponent adjustments to a metric.
        
        Args:
            df: DataFrame with team metrics
            metric: Name of metric to adjust
            opponent_strength: DataFrame with opponent strength metrics
            matchup_adjustments: DataFrame with matchup adjustments
            
        Returns:
            Series with adjusted metrics
        """
        # Get base metric value
        base_value = df[metric]
        
        # Calculate total adjustment
        total_adjustment = matchup_adjustments.sum(axis=1)
        
        # Apply adjustment
        adjusted_value = base_value * (1 + total_adjustment)
        
        return adjusted_value
    
    def calculate_adjusted_metrics(
        self,
        offensive_edp: pd.DataFrame,
        defensive_edp: pd.DataFrame
    ) -> Dict[str, pd.Series]:
        """
        Calculate opponent-adjusted EDP metrics.
        
        Args:
            offensive_edp: DataFrame with offensive EDP metrics
            defensive_edp: DataFrame with defensive EDP metrics
            
        Returns:
            Dictionary with adjusted offensive and defensive metrics
        """
        # Combine metrics
        combined_df = pd.merge(
            offensive_edp,
            defensive_edp,
            on=['team', 'game_id'],
            how='outer',
            validate='1:1'
        )
        
        # Calculate opponent strength metrics
        opponent_strength = self.calculate_opponent_strength_metrics(
            combined_df,
            metrics=self.settings['metrics'],
            window=self.settings['window']
        )
        
        # Calculate matchup adjustments
        matchup_adjustments = self.calculate_matchup_adjustments(
            combined_df[['offensive_edp', 'defensive_edp']],
            opponent_strength,
            weights=self.settings['weights']
        )
        
        # Apply adjustments
        adjusted_offensive = self.apply_opponent_adjustments(
            combined_df,
            'offensive_edp',
            opponent_strength,
            matchup_adjustments
        )
        
        adjusted_defensive = self.apply_opponent_adjustments(
            combined_df,
            'defensive_edp',
            opponent_strength,
            matchup_adjustments
        )
        
        return {
            'offensive': adjusted_offensive,
            'defensive': adjusted_defensive
        } 