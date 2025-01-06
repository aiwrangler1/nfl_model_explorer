"""Simple EDP calculator for NFL drive analysis."""

import pandas as pd
import numpy as np

class EDPCalculator:
    """Calculates Expected Drive Points (EDP) metrics."""
    
    def calculate_drive_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate drive-level metrics."""
        # Create unique drive identifier
        df = df.copy()
        df['drive_id'] = df['game_id'] + '_' + df['drive'].astype(str)
        
        # Calculate drive-level EPA sum
        drive_metrics = df.groupby(['drive_id', 'posteam', 'defteam', 'game_id', 'week']).agg({
            'epa': 'sum',
            'play_id': 'count'
        }).reset_index()
        
        # Rename columns
        drive_metrics = drive_metrics.rename(columns={
            'epa': 'drive_points',
            'play_id': 'play_count'
        })
        
        return drive_metrics
    
    def calculate_team_metrics(self, drive_metrics: pd.DataFrame) -> tuple:
        """Calculate team-level offensive and defensive metrics."""
        # Offensive metrics
        offensive = drive_metrics.groupby(['posteam', 'game_id', 'week']).agg({
            'drive_points': 'sum',
            'drive_id': 'count'
        }).reset_index()
        
        offensive = offensive.rename(columns={
            'posteam': 'team',
            'drive_points': 'earned_drive_points_off',
            'drive_id': 'drive_count_off'
        })
        
        # Defensive metrics
        defensive = drive_metrics.groupby(['defteam', 'game_id', 'week']).agg({
            'drive_points': 'sum',
            'drive_id': 'count'
        }).reset_index()
        
        defensive = defensive.rename(columns={
            'defteam': 'team',
            'drive_points': 'earned_drive_points_def',
            'drive_id': 'drive_count_def'
        })
        
        return offensive, defensive

    def calculate_strength_adjusted_edp(self, offensive: pd.DataFrame, defensive: pd.DataFrame, iterations: int = 20) -> pd.DataFrame:
        """
        Calculates strength-of-schedule-adjusted EDP using an iterative process.
        Each iteration adjusts team performances based on opponent quality.
        
        Args:
            offensive: DataFrame with raw offensive EDP metrics
            defensive: DataFrame with raw defensive EDP metrics
            iterations: Number of iterations for adjustment (default: 20)
        
        Returns:
            DataFrame with strength-adjusted offensive and defensive EDP
        """
        # Merge offensive and defensive data
        team_metrics = pd.merge(offensive, defensive, on=['team', 'game_id', 'week'], how='outer')
        
        # Initialize adjusted EDP columns with raw metrics
        team_metrics['adjusted_edp_off'] = team_metrics['earned_drive_points_off']
        team_metrics['adjusted_edp_def'] = team_metrics['earned_drive_points_def']
        
        # Store original means for final scaling
        original_off_mean = team_metrics['earned_drive_points_off'].mean()
        original_def_mean = team_metrics['earned_drive_points_def'].mean()
        
        for _ in range(iterations):
            # Create dictionaries for quick lookup of current adjusted EDPs
            off_edp_dict = team_metrics.set_index('team')['adjusted_edp_off'].to_dict()
            def_edp_dict = team_metrics.set_index('team')['adjusted_edp_def'].to_dict()
            
            # Calculate current means for relative adjustments
            current_off_mean = team_metrics['adjusted_edp_off'].mean()
            current_def_mean = team_metrics['adjusted_edp_def'].mean()
            
            # Iterate through each game and adjust EDP based on opponent's strength
            new_off_edp = []
            new_def_edp = []
            
            for _, row in team_metrics.iterrows():
                opponent = self.get_opponent(row['team'], row['game_id'], team_metrics)
                
                if opponent is None:
                    new_off_edp.append(row['adjusted_edp_off'])
                    new_def_edp.append(row['adjusted_edp_def'])
                    continue
                
                # Get opponent's current adjusted metrics
                opponent_def_edp = def_edp_dict.get(opponent, current_def_mean)
                opponent_off_edp = off_edp_dict.get(opponent, current_off_mean)
                
                # Calculate adjustments based on opponent strength relative to mean
                off_adjustment = (current_def_mean - opponent_def_edp) * 0.33
                def_adjustment = (current_off_mean - opponent_off_edp) * 0.33
                
                # Apply adjustments to raw metrics
                adjusted_off_edp = row['earned_drive_points_off'] + off_adjustment
                adjusted_def_edp = row['earned_drive_points_def'] + def_adjustment
                
                new_off_edp.append(adjusted_off_edp)
                new_def_edp.append(adjusted_def_edp)
            
            # Update adjusted EDPs
            team_metrics['adjusted_edp_off'] = new_off_edp
            team_metrics['adjusted_edp_def'] = new_def_edp
            
            # Re-center around original means to maintain scale
            team_metrics['adjusted_edp_off'] = (team_metrics['adjusted_edp_off'] - 
                team_metrics['adjusted_edp_off'].mean() + original_off_mean)
            team_metrics['adjusted_edp_def'] = (team_metrics['adjusted_edp_def'] - 
                team_metrics['adjusted_edp_def'].mean() + original_def_mean)
        
        return team_metrics
    
    def get_opponent(self, team: str, game_id: str, team_metrics: pd.DataFrame) -> str:
        """Helper function to get opponent in a given game."""
        opponent_row = team_metrics[
            (team_metrics['game_id'] == game_id) & 
            (team_metrics['team'] != team)
        ]
        if not opponent_row.empty:
            return opponent_row['team'].iloc[0]
        return None  # Occurs when merging data 