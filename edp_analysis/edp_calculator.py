"""
Functions for calculating Expected Defensive Points (EDP) metrics.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from .config import (
    EDP_COEFFICIENTS,
    EXPLOSIVE_PLAY_THRESHOLD,
    SHORT_EDP_WINDOW,
    LONG_EDP_WINDOW,
    EDP_DECAY_FACTOR,
    OPPONENT_STRENGTH_SETTINGS
)
from .opponent_strength import OpponentStrengthAdjuster

class EDPCalculator:
    def __init__(self):
        """Initialize the EDP calculator with default settings."""
        self.short_window = SHORT_EDP_WINDOW
        self.long_window = LONG_EDP_WINDOW
        self.decay_factor = EDP_DECAY_FACTOR
        
    def load_and_process_data(self, seasons=None) -> pd.DataFrame:
        """
        Load and process NFL play-by-play data.
        
        Args:
            seasons: List of seasons to load. If None, loads current season.
            
        Returns:
            pd.DataFrame: Processed play-by-play data
        """
        if seasons is None:
            seasons = [2023]
            
        # Load play-by-play data
        df = nfl.import_pbp_data(seasons)
        
        # Basic filtering
        df = df[
            (df['play_type'].isin(['run', 'pass', 'field_goal'])) &
            (~df['down'].isna())
        ].copy()
        
        return self.prepare_play_data(df)
    
    def calculate_success(self, play_type, down, yards_gained, ydstogo, field_goal_result=None):
        """
        Calculate if a play was successful based on type and situation.
        
        Args:
            play_type (str): Type of play (run, pass, field_goal)
            down (int): Current down
            yards_gained (float): Yards gained on the play
            ydstogo (float): Yards needed for first down
            field_goal_result (str, optional): Result of field goal attempt
        
        Returns:
            int: 1 if play was successful, 0 otherwise
        """
        try:
            if play_type == 'field_goal':
                return 1 if field_goal_result == 'made' else 0
            elif down == 1:
                return int(yards_gained >= 0.4 * ydstogo)
            elif down == 2:
                return int(yards_gained >= 0.6 * ydstogo)
            elif down in [3, 4]:
                return int(yards_gained >= ydstogo)
            else:
                return 0
        except Exception:
            return 0

    def calculate_play_success(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate play success based on down, distance, and result.
        
        Args:
            df (pd.DataFrame): The play-by-play dataframe.
            
        Returns:
            pd.DataFrame: The input dataframe with a 'success' column added.
        """
        df['success'] = df.apply(
            lambda x: self.calculate_success(
                x['play_type'], 
                x['down'], 
                x['yards_gained'], 
                x['ydstogo'],
                x.get('field_goal_result')
            ), 
            axis=1
        )
        return df

    def calculate_drive_success_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate the success rate for each drive.
        
        Args:
            df (pd.DataFrame): The play-by-play dataframe with a 'success' column.
            
        Returns:
            pd.DataFrame: A dataframe with drive success rates.
        """
        drive_success = df.groupby(['game_id', 'drive']).agg(
            drive_plays=('play_id', 'count'),
            drive_success_plays=('success', 'sum')
        )
        drive_success['drive_success_rate'] = drive_success['drive_success_plays'] / drive_success['drive_plays']
        return drive_success.reset_index()

    def calculate_drive_level_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate drive-level metrics.
        
        Args:
            df (pd.DataFrame): The play-by-play dataframe.
            
        Returns:
            pd.DataFrame: A dataframe with drive-level metrics.
        """
        drive_metrics = df.groupby(['game_id', 'posteam', 'drive', 'season', 'week']).agg(
            drive_plays=('play_id', 'count'),
            drive_points=('points', 'sum'),
            num_explosive_plays=('explosive_play', 'sum'),
            num_late_down_conversions=('late_down_conversion', 'sum'),
            scoring_zone_successes=('scoring_zone_success', 'sum'),
            entered_red_zone=('is_scoring_opportunity', 'max'),
            drive_success_plays=('success', 'sum'),
            drive_epa=('epa', 'sum')
        ).reset_index()
        
        # Calculate drive success rate
        drive_metrics['drive_success_rate'] = (
            drive_metrics['drive_success_plays'] / drive_metrics['drive_plays']
        )
        
        # Calculate red zone failures (entered red zone but didn't score)
        drive_metrics['red_zone_failure'] = (
            (drive_metrics['entered_red_zone'] == 1) & 
            (drive_metrics['drive_points'] == 0)
        ).astype(int)
        
        return drive_metrics

    def calculate_raw_edp(self, drive_metrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate raw EDP from drive metrics.
        
        Args:
            drive_metrics (pd.DataFrame): A dataframe with drive-level metrics.
            
        Returns:
            pd.DataFrame: The input dataframe with a 'raw_edp' column added.
        """
        drive_metrics['raw_edp'] = (
            drive_metrics['drive_epa'] * EDP_COEFFICIENTS['drive_epa'] +  
            drive_metrics['drive_success_rate'] * EDP_COEFFICIENTS['drive_success_rate'] +  
            drive_metrics['num_explosive_plays'] * EDP_COEFFICIENTS['num_explosive_plays'] +  
            drive_metrics['num_late_down_conversions'] * EDP_COEFFICIENTS['num_late_down_conversions'] +  
            drive_metrics['scoring_zone_successes'] * EDP_COEFFICIENTS['scoring_zone_successes'] -
            drive_metrics['red_zone_failure'] * abs(EDP_COEFFICIENTS['red_zone_failure'])
        )
        return drive_metrics

    def prepare_play_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare play-by-play data for EDP calculation by deriving necessary columns.
        
        Args:
            df (pd.DataFrame): Raw play-by-play data
            
        Returns:
            pd.DataFrame: Processed data with all required columns
        """
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Calculate points from scoring plays
        df['points'] = 0
        df.loc[df['touchdown'] == 1, 'points'] = 6
        df.loc[df['field_goal_result'] == 'made', 'points'] = 3
        df.loc[df['extra_point_result'] == 'good', 'points'] = 1
        df.loc[df['two_point_conv_result'] == 'success', 'points'] = 2
        df.loc[df['safety'] == 1, 'points'] = 2
        
        # Define scoring opportunities
        df['is_scoring_opportunity'] = (
            (df['yardline_100'] <= 20) | 
            (df['field_goal_attempt'] == 1)
        ).astype(int)
        
        # Calculate success for each play
        df = self.calculate_play_success(df)
        
        # Calculate explosive plays
        df['explosive_play'] = (df['yards_gained'] >= EXPLOSIVE_PLAY_THRESHOLD).astype(int)
        
        # Calculate late down conversions
        df['late_down_conversion'] = ((df['down'] >= 3) & (df['success'] == 1)).astype(int)
        
        # Calculate scoring zone successes
        df['scoring_zone_success'] = (df['is_scoring_opportunity'] & df['success']).astype(int)
        
        # Calculate EPA if not already present
        if 'epa' not in df.columns:
            df['epa'] = df.apply(self.calculate_epa, axis=1)
        
        return df
    
    def calculate_epa(self, play: pd.Series) -> float:
        """
        Calculate Expected Points Added (EPA) for a play.
        
        Args:
            play: Series containing play information
            
        Returns:
            float: EPA value for the play
        """
        try:
            # Base EPA calculation
            if play['play_type'] == 'field_goal':
                if play['field_goal_result'] == 'made':
                    return 3 - self.calculate_field_goal_ep(play['yardline_100'])
                else:
                    return -self.calculate_field_goal_ep(play['yardline_100'])
            
            # For normal plays
            start_ep = self.calculate_expected_points(
                play['yardline_100'],
                play['down'],
                play['ydstogo']
            )
            
            # Handle scoring plays
            if play['points'] > 0:
                return play['points'] - start_ep
            
            # Handle turnovers
            if play.get('interception') == 1 or play.get('fumble_lost') == 1:
                end_ep = -self.calculate_expected_points(
                    100 - play['yardline_100'] - play.get('yards_gained', 0),
                    1,
                    10
                )
            else:
                # Normal play
                yards_gained = play.get('yards_gained', 0)
                new_yards_to_go = max(play['ydstogo'] - yards_gained, 0)
                new_down = play['down'] + 1 if new_yards_to_go > 0 else 1
                if new_down > 4:
                    end_ep = -self.calculate_expected_points(
                        100 - play['yardline_100'],
                        1,
                        10
                    )
                else:
                    end_ep = self.calculate_expected_points(
                        max(0, play['yardline_100'] - yards_gained),
                        new_down if new_yards_to_go > 0 else 1,
                        10 if new_yards_to_go <= 0 else new_yards_to_go
                    )
            
            return end_ep - start_ep
        except Exception:
            return 0.0
    
    def calculate_expected_points(self, yards_to_goal: float, down: int, yards_to_go: float) -> float:
        """
        Calculate expected points for a given situation.
        
        Args:
            yards_to_goal: Yards to the goal line
            down: Current down (1-4)
            yards_to_go: Yards needed for first down
            
        Returns:
            float: Expected points
        """
        # Simple expected points model based on field position
        if yards_to_goal <= 20:
            return 4.0  # High scoring probability in red zone
        elif yards_to_goal <= 40:
            return 3.0  # Field goal range
        elif yards_to_goal <= 60:
            return 2.0  # Good field position
        elif yards_to_goal <= 80:
            return 1.0  # Middle of field
        else:
            return 0.5  # Deep in own territory
    
    def calculate_field_goal_ep(self, yards_to_goal: float) -> float:
        """
        Calculate expected points for a field goal attempt.
        
        Args:
            yards_to_goal: Yards to the goal line
            
        Returns:
            float: Expected points
        """
        # Simple field goal probability model
        if yards_to_goal <= 20:
            return 2.7  # Very makeable
        elif yards_to_goal <= 30:
            return 2.4  # Still high probability
        elif yards_to_goal <= 40:
            return 2.1  # Moderate difficulty
        elif yards_to_goal <= 50:
            return 1.5  # Getting difficult
        else:
            return 0.9  # Very difficult

    def calculate_offensive_edp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate offensive EDP metrics for teams.
        
        Args:
            df (pd.DataFrame): The play-by-play dataframe.
            
        Returns:
            pd.DataFrame: A dataframe with team-level offensive EDP metrics.
        """
        # Calculate drive-level metrics
        drive_metrics = self.calculate_drive_level_metrics(df)
        drive_metrics = self.calculate_raw_edp(drive_metrics)
        
        # Calculate offensive EDP
        offensive_edp = drive_metrics.groupby(['posteam', 'game_id', 'season', 'week']).agg(
            offensive_edp=('raw_edp', 'sum')
        ).reset_index()
        
        offensive_edp = offensive_edp.rename(columns={'posteam': 'team'})
        return offensive_edp

    def calculate_defensive_edp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate defensive EDP metrics for teams.
        
        Args:
            df (pd.DataFrame): The play-by-play dataframe.
            
        Returns:
            pd.DataFrame: A dataframe with team-level defensive EDP metrics.
        """
        # Swap offense/defense for defensive calculation
        defensive_df = df.copy()
        defensive_df = defensive_df.rename(columns={'posteam': 'temp_team', 'defteam': 'posteam'})
        defensive_df['defteam'] = defensive_df['temp_team']
        defensive_df = defensive_df.drop('temp_team', axis=1)
        
        # Calculate drive-level metrics
        defensive_metrics = self.calculate_drive_level_metrics(defensive_df)
        defensive_metrics['raw_edp'] = -defensive_metrics['drive_epa']
        
        # Calculate defensive EDP
        defensive_edp = defensive_metrics.groupby(['posteam', 'game_id', 'season', 'week']).agg(
            defensive_edp=('raw_edp', 'sum')
        ).reset_index()
        
        defensive_edp = defensive_edp.rename(columns={'posteam': 'team'})
        return defensive_edp

    def calculate_team_edp(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate both offensive and defensive EDP metrics for teams.
        
        Args:
            df (pd.DataFrame): The play-by-play dataframe.
            
        Returns:
            pd.DataFrame: A dataframe with team-level EDP metrics.
        """
        # Calculate offensive and defensive EDP
        offensive_edp = self.calculate_offensive_edp(df)
        defensive_edp = self.calculate_defensive_edp(df)
        
        # Combine metrics
        combined_edp = pd.merge(
            offensive_edp, 
            defensive_edp, 
            on=['team', 'game_id'],
            how='outer',
            validate='1:1'
        )
        
        return combined_edp