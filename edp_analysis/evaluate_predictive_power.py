"""
Evaluation of EDP's predictive power across multiple dimensions.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from typing import Dict, Tuple

def calculate_win_probability_metrics(
    df: pd.DataFrame,
    edp_col: str = 'adjusted_offensive_edp'
) -> Dict[str, float]:
    """
    Calculate how well EDP predicts wins.
    
    Args:
        df: DataFrame with EDP and game outcomes
        edp_col: Column name for EDP metric to evaluate
        
    Returns:
        Dictionary of predictive metrics
    """
    # Calculate EDP differential for each game
    game_metrics = []
    for game_id in df['game_id'].unique():
        game_df = df[df['game_id'] == game_id]
        if len(game_df) == 2:  # Ensure we have both teams
            home_team = game_df[game_df['home_away'] == 'home'].iloc[0]
            away_team = game_df[game_df['home_away'] == 'away'].iloc[0]
            
            edp_diff = home_team[edp_col] - away_team[edp_col]
            point_diff = home_team['points'] - away_team['points']
            
            game_metrics.append({
                'game_id': game_id,
                'season': home_team['season'],
                'week': home_team['week'],
                'edp_differential': edp_diff,
                'point_differential': point_diff,
                'home_win': int(point_diff > 0)
            })
    
    game_metrics = pd.DataFrame(game_metrics)
    
    # Calculate predictive metrics
    metrics = {
        'win_probability_auc': roc_auc_score(
            game_metrics['home_win'],
            game_metrics['edp_differential']
        ),
        'win_probability_log_loss': log_loss(
            game_metrics['home_win'],
            1 / (1 + np.exp(-game_metrics['edp_differential'] / 100))  # Simple logistic transform
        )
    }
    
    return metrics

def evaluate_drive_prediction(
    df: pd.DataFrame,
    edp_col: str = 'adjusted_offensive_edp',
    future_window: int = 3
) -> Dict[str, float]:
    """
    Evaluate how well EDP predicts future drive success.
    
    Args:
        df: DataFrame with drive-level data
        edp_col: Column name for EDP metric to evaluate
        future_window: Number of future drives to predict
        
    Returns:
        Dictionary of predictive metrics
    """
    # Calculate rolling drive success metrics
    df = df.sort_values(['team', 'season', 'week', 'drive'])
    df['future_drive_success'] = df.groupby('team')['drive_success_rate'].shift(-future_window)
    df['future_points_per_drive'] = df.groupby('team')['points'].shift(-future_window)
    
    # Remove rows where we don't have future data
    df = df.dropna(subset=['future_drive_success', 'future_points_per_drive'])
    
    metrics = {
        'future_drive_success_correlation': df[edp_col].corr(df['future_drive_success']),
        'future_points_correlation': df[edp_col].corr(df['future_points_per_drive'])
    }
    
    return metrics

def evaluate_season_prediction(
    df: pd.DataFrame,
    edp_col: str = 'adjusted_offensive_edp',
    early_week_cutoff: int = 8
) -> Dict[str, float]:
    """
    Evaluate how well early-season EDP predicts late-season performance.
    
    Args:
        df: DataFrame with game-level data
        edp_col: Column name for EDP metric to evaluate
        early_week_cutoff: Week to split early/late season
        
    Returns:
        Dictionary of predictive metrics
    """
    # Calculate early and late season metrics
    early_season = df[df['week'] <= early_week_cutoff].groupby(['season', 'team'])[edp_col].mean()
    late_season = df[df['week'] > early_week_cutoff].groupby(['season', 'team'])[edp_col].mean()
    
    # Calculate win percentages
    early_wins = df[df['week'] <= early_week_cutoff].groupby(['season', 'team'])['won'].mean()
    late_wins = df[df['week'] > early_week_cutoff].groupby(['season', 'team'])['won'].mean()
    
    metrics = {
        'early_late_edp_correlation': early_season.corr(late_season),
        'early_edp_late_wins_correlation': early_season.corr(late_wins),
        'early_wins_late_wins_correlation': early_wins.corr(late_wins)  # Baseline comparison
    }
    
    return metrics

def print_evaluation_summary(metrics: Dict[str, Dict[str, float]]) -> None:
    """Print a formatted summary of all predictive metrics."""
    print("\nEDP Predictive Power Evaluation")
    print("=" * 50)
    
    print("\n1. Win Probability Prediction")
    print("-" * 30)
    print(f"AUC-ROC: {metrics['win_prob']['win_probability_auc']:.3f}")
    print(f"Log Loss: {metrics['win_prob']['win_probability_log_loss']:.3f}")
    
    print("\n2. Future Drive Prediction")
    print("-" * 30)
    print(f"Drive Success Correlation: {metrics['drive_pred']['future_drive_success_correlation']:.3f}")
    print(f"Points Correlation: {metrics['drive_pred']['future_points_correlation']:.3f}")
    
    print("\n3. Season-Level Prediction")
    print("-" * 30)
    print(f"Early-Late EDP Correlation: {metrics['season_pred']['early_late_edp_correlation']:.3f}")
    print(f"Early EDP → Late Wins: {metrics['season_pred']['early_edp_late_wins_correlation']:.3f}")
    print(f"(Baseline) Early-Late Wins: {metrics['season_pred']['early_wins_late_wins_correlation']:.3f}")

def evaluate_edp_predictive_power(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Comprehensive evaluation of EDP's predictive power.
    
    Args:
        df: DataFrame with all required game and drive level data
        
    Returns:
        Dictionary of all predictive metrics
    """
    metrics = {
        'win_prob': calculate_win_probability_metrics(df),
        'drive_pred': evaluate_drive_prediction(df),
        'season_pred': evaluate_season_prediction(df)
    }
    
    print_evaluation_summary(metrics)
    return metrics 