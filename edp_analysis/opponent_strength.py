"""
Opponent strength adjustment calculations for EDP metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from .config import OPPONENT_STRENGTH_SETTINGS
import logging

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
        """Calculate opponent-adjusted metrics."""
        try:
            # Create combined dataframe for calculations
            combined_df = pd.merge(
                offensive_edp,
                defensive_edp,
                on=['team', 'game_id', 'season', 'week'],
                suffixes=('_off', '_def')
            )
            
            # Calculate league averages
            league_avg_off = combined_df['offensive_edp'].mean()
            league_avg_def = combined_df['defensive_edp'].mean()
            
            # Initialize dictionaries for adjusted values
            adjusted_offensive = {}
            adjusted_defensive = {}
            
            # Calculate opponent strength for each team
            opponent_strength = self._calculate_opponent_strength(combined_df)
            
            # Calculate adjustments
            for team in combined_df['team'].unique():
                team_data = combined_df[combined_df['team'] == team]
                if len(team_data) > 0:
                    # Offensive adjustment
                    raw_off_edp = team_data['offensive_edp'].mean()
                    opp_def_strength = opponent_strength.get(team, {}).get('defensive', league_avg_def)
                    adjusted_offensive[team] = raw_off_edp + (league_avg_def - opp_def_strength)
                    
                    # Defensive adjustment
                    raw_def_edp = team_data['defensive_edp'].mean()
                    opp_off_strength = opponent_strength.get(team, {}).get('offensive', league_avg_off)
                    adjusted_defensive[team] = raw_def_edp + (league_avg_off - opp_off_strength)
            
            # Fill any missing teams with league averages
            all_teams = set(offensive_edp['team'].unique()) | set(defensive_edp['team'].unique())
            for team in all_teams:
                if team not in adjusted_offensive:
                    adjusted_offensive[team] = league_avg_off
                if team not in adjusted_defensive:
                    adjusted_defensive[team] = league_avg_def
            
            logging.info(f"Adjusted metrics calculated for {len(adjusted_offensive)} teams")
            return {
                'offensive': adjusted_offensive,
                'defensive': adjusted_defensive
            }
            
        except Exception as e:
            logging.error(f"Error calculating adjusted metrics: {str(e)}")
            raise 
    
    def _calculate_opponent_strength(self, combined_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calculate opponent strength metrics for each team.
        
        Args:
            combined_df: DataFrame with offensive and defensive metrics
        
        Returns:
            Dictionary mapping teams to their opponent strength metrics
        """
        opponent_strength = {}
        
        try:
            # Calculate league averages for reference
            league_avg_off = combined_df['offensive_edp'].mean()
            league_avg_def = combined_df['defensive_edp'].mean()
            
            # For each team, calculate average opponent metrics
            for team in combined_df['team'].unique():
                team_games = combined_df[combined_df['team'] == team]
                
                # Get list of opponents
                opponents = team_games['opponent'].unique() if 'opponent' in team_games.columns else []
                
                if len(opponents) > 0:
                    # Calculate average opponent offensive and defensive metrics
                    opponent_metrics = combined_df[combined_df['team'].isin(opponents)]
                    
                    opponent_strength[team] = {
                        'offensive': opponent_metrics['offensive_edp'].mean(),
                        'defensive': opponent_metrics['defensive_edp'].mean()
                    }
                else:
                    # If no opponent data, use league averages
                    opponent_strength[team] = {
                        'offensive': league_avg_off,
                        'defensive': league_avg_def
                    }
            
            logging.info(f"Calculated opponent strength metrics for {len(opponent_strength)} teams")
            return opponent_strength
            
        except Exception as e:
            logging.error(f"Error calculating opponent strength: {str(e)}")
            # Return empty dict if calculation fails
            return {team: {'offensive': league_avg_off, 'defensive': league_avg_def} 
                    for team in combined_df['team'].unique()}