"""
Utility functions for processing NFL data.
"""

import pandas as pd
from typing import List, Dict

def standardize_team_names(
    df: pd.DataFrame,
    team_cols: List[str],
    team_map: Dict[str, str]
) -> pd.DataFrame:
    """
    Standardize team names in a dataframe using a mapping dictionary.
    
    Args:
        df (pd.DataFrame): The dataframe to process.
        team_cols (List[str]): The list of column names containing team names to standardize.
        team_map (Dict[str, str]): A dictionary mapping non-standard team names to standard names.
        
    Returns:
        pd.DataFrame: The dataframe with standardized team names.
    """
    df = df.copy()
    for col in team_cols:
        if col in df.columns:
            df[col] = df[col].map(team_map).fillna(df[col])
    return df

def calculate_drive_metrics(
    df: pd.DataFrame,
    group_cols: List[str] = ['game_id', 'drive']
) -> pd.DataFrame:
    """
    Calculate drive-level metrics from play-by-play data.
    
    Args:
        df (pd.DataFrame): The play-by-play dataframe to process.
        group_cols (List[str], optional): The columns to group by for drive calculations. Defaults to ['game_id', 'drive'].
        
    Returns:
        pd.DataFrame: A dataframe with drive-level metrics.
    """
    metrics = df.groupby(group_cols).agg(
        drive_epa=('epa', 'sum'),
        num_plays=('play_id', 'count'),
        num_successful_plays=('success', 'sum'),
        num_explosive_plays=('explosive_play', 'sum'),
        num_scoring_zone_successes=('scoring_zone_success', 'sum')
    ).reset_index()
    
    metrics['drive_success_rate'] = metrics['num_successful_plays'] / metrics['num_plays']
    
    return metrics

def calculate_game_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate game-level metrics from play-by-play data.
    
    Args:
        df (pd.DataFrame): The play-by-play dataframe to process.
        
    Returns:
        pd.DataFrame: A dataframe with game-level metrics.
    """
    metrics = df.groupby(['game_id', 'posteam']).agg(
        game_avg_epa=('epa', 'mean'),
        game_total_epa=('epa', 'sum'),
        game_epa_volatility=('epa', 'std'),
        game_avg_yards=('yards_gained', 'mean'),
        game_total_yards=('yards_gained', 'sum'),
        game_yards_volatility=('yards_gained', 'std'),
        game_fg_attempts=('field_goal_attempt', 'sum'),
        game_fg_made=('field_goal_result', lambda x: (x == 'made').sum()),
        game_total_plays=('play_id', 'count'),
        final_score_differential=('score_differential', 'last')
    ).reset_index()
    
    metrics['game_fg_success_rate'] = metrics['game_fg_made'] / metrics['game_fg_attempts'].clip(lower=1)
    metrics['yards_per_play'] = metrics['game_total_yards'] / metrics['game_total_plays']
    metrics['epa_per_play'] = metrics['game_total_epa'] / metrics['game_total_plays']
    
    return metrics