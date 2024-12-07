"""
Validation script to compare original and opponent-adjusted EDP models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import nfl_data_py as nfl
from datetime import datetime

from edp_analysis.edp_calculator import calculate_team_edp, calculate_rolling_edp, prepare_play_data
from edp_analysis.opponent_strength import (
    calculate_opponent_strength_metrics,
    calculate_matchup_adjustments,
    apply_opponent_adjustments
)
from edp_analysis.config import OPPONENT_STRENGTH_SETTINGS

def load_historical_data(start_year: int = 2013) -> pd.DataFrame:
    """
    Load NFL play-by-play data from nfl_data_py for validation.
    
    Args:
        start_year (int): First season to include in analysis
        
    Returns:
        pd.DataFrame: Historical play-by-play data
    """
    current_year = datetime.now().year
    years = list(range(start_year, current_year + 1))
    
    print(f"Loading data for years {start_year}-{current_year}...")
    pbp_data = nfl.import_pbp_data(years)
    
    # Filter for regular season games
    pbp_data = pbp_data[pbp_data['season_type'] == 'REG']
    
    # Print data info for debugging
    print("\nDataset Info:")
    print("-" * 50)
    print(f"Number of plays: {len(pbp_data):,}")
    print(f"Years covered: {pbp_data['season'].min()} - {pbp_data['season'].max()}")
    print("\nKey columns present:")
    required_columns = [
        # Game context
        'game_id', 'posteam', 'defteam', 'home_team', 'away_team', 'season',
        # Play details
        'play_type', 'down', 'yards_gained', 'ydstogo', 'yardline_100',
        'field_goal_attempt', 'field_goal_result', 'drive', 'epa',
        # Scoring
        'total_home_score', 'total_away_score'
    ]
    for col in required_columns:
        present = col in pbp_data.columns
        print(f"- {col}: {'✓' if present else '✗'}")
        if not present:
            print(f"  WARNING: Required column '{col}' is missing!")
    
    return pbp_data

def calculate_game_points(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate actual points scored per game by each team.
    
    Args:
        df (pd.DataFrame): Play-by-play data
        
    Returns:
        pd.DataFrame: Game-level points by team, with proper temporal ordering
    """
    # Get final scores for each game
    game_scores = df.groupby(['game_id', 'season', 'week']).agg({
        'total_home_score': 'max',
        'total_away_score': 'max',
        'home_team': 'first',
        'away_team': 'first'
    }).reset_index()
    
    # Create two rows per game, one for each team
    home_points = game_scores[['game_id', 'home_team', 'total_home_score', 'week', 'season']].rename(
        columns={'home_team': 'team', 'total_home_score': 'points'}
    )
    away_points = game_scores[['game_id', 'away_team', 'total_away_score', 'week', 'season']].rename(
        columns={'away_team': 'team', 'total_away_score': 'points'}
    )
    
    # Combine and sort by temporal order
    game_points = pd.concat([home_points, away_points], ignore_index=True)
    game_points = game_points.sort_values(['season', 'week', 'game_id'])
    
    return game_points

def calculate_model_metrics(
    df: pd.DataFrame,
    original_edp: pd.DataFrame,
    adjusted_edp: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate comparative metrics between original and adjusted EDP models.
    
    Args:
        df (pd.DataFrame): Source play-by-play data
        original_edp (pd.DataFrame): Original EDP calculations
        adjusted_edp (pd.DataFrame): Opponent-adjusted EDP calculations
        
    Returns:
        Dict[str, float]: Dictionary of comparative metrics
    """
    metrics = {}
    
    # Calculate actual points scored with temporal ordering
    game_points = calculate_game_points(df)
    
    # Add temporal ordering to EDP dataframes if not present
    if 'week' not in original_edp.columns or 'season' not in original_edp.columns:
        game_info = df.groupby('game_id').agg({
            'week': 'first',
            'season': 'first'
        }).reset_index()
        original_edp = pd.merge(original_edp, game_info, on='game_id')
        adjusted_edp = pd.merge(adjusted_edp, game_info, on='game_id')
    
    # Sort all dataframes by temporal order
    for df_to_sort in [game_points, original_edp, adjusted_edp]:
        df_to_sort.sort_values(['season', 'week', 'game_id'], inplace=True)
    
    # Calculate week-over-week stability
    def calculate_wow_stability(df: pd.DataFrame, metric_col: str) -> float:
        # Create lagged values for each team within each season
        df = df.copy()
        df['prev_value'] = df.groupby(['team', 'season'])[metric_col].shift(1)
        # Remove rows with NaN (first game for each team in each season)
        df = df.dropna(subset=['prev_value'])
        # Calculate correlation
        return df[metric_col].corr(df['prev_value'])
    
    metrics['original_wow_stability'] = calculate_wow_stability(
        original_edp, 'weighted_offensive_edp'
    )
    metrics['adjusted_wow_stability'] = calculate_wow_stability(
        adjusted_edp, 'adjusted_offensive_edp'
    )
    
    # Merge actual points with EDP predictions, maintaining temporal order
    original_comparison = pd.merge(
        game_points,
        original_edp[['game_id', 'team', 'weighted_offensive_edp', 'week', 'season']],
        on=['game_id', 'team', 'week', 'season']
    )
    
    adjusted_comparison = pd.merge(
        game_points,
        adjusted_edp[['game_id', 'team', 'adjusted_offensive_edp', 'week', 'season']],
        on=['game_id', 'team', 'week', 'season']
    )
    
    # Print some diagnostic information
    print("\nData Validation:")
    print("-" * 50)
    print(f"Total games analyzed: {len(game_points) // 2}")
    print(f"Unique teams: {len(game_points['team'].unique())}")
    print(f"Date range: {game_points['season'].min()}-{game_points['season'].max()}")
    print(f"\nSample of point comparisons:")
    print(original_comparison[['team', 'season', 'week', 'points', 'weighted_offensive_edp']].head())
    
    # Points prediction accuracy
    metrics['original_rmse'] = np.sqrt(mean_squared_error(
        original_comparison['points'],
        original_comparison['weighted_offensive_edp']
    ))
    
    metrics['adjusted_rmse'] = np.sqrt(mean_squared_error(
        adjusted_comparison['points'],
        adjusted_comparison['adjusted_offensive_edp']
    ))
    
    # R-squared for points prediction
    metrics['original_r2'] = r2_score(
        original_comparison['points'],
        original_comparison['weighted_offensive_edp']
    )
    
    metrics['adjusted_r2'] = r2_score(
        adjusted_comparison['points'],
        adjusted_comparison['adjusted_offensive_edp']
    )
    
    return metrics

def plot_model_comparison(
    original_edp: pd.DataFrame,
    adjusted_edp: pd.DataFrame,
    metrics: Dict[str, float]
) -> None:
    """
    Create visualization comparing original and adjusted EDP models.
    
    Args:
        original_edp (pd.DataFrame): Original EDP calculations
        adjusted_edp (pd.DataFrame): Opponent-adjusted EDP calculations
        metrics (Dict[str, float]): Comparative metrics
    """
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Distribution of EDP values
    plt.subplot(2, 2, 1)
    sns.kdeplot(data=original_edp['weighted_offensive_edp'], label='Original EDP')
    sns.kdeplot(data=adjusted_edp['adjusted_offensive_edp'], label='Adjusted EDP')
    plt.title('Distribution of EDP Values')
    plt.legend()
    
    # Plot 2: Week-over-week stability
    plt.subplot(2, 2, 2)
    stability_data = pd.DataFrame({
        'Original': [metrics['original_wow_stability']],
        'Adjusted': [metrics['adjusted_wow_stability']]
    }).melt()
    sns.barplot(data=stability_data, x='variable', y='value')
    plt.title('Week-over-Week Stability')
    plt.ylim(0, 1)
    
    # Plot 3: RMSE Comparison
    plt.subplot(2, 2, 3)
    rmse_data = pd.DataFrame({
        'Original': [metrics['original_rmse']],
        'Adjusted': [metrics['adjusted_rmse']]
    }).melt()
    sns.barplot(data=rmse_data, x='variable', y='value')
    plt.title('RMSE in Points Prediction')
    
    # Plot 4: R-squared Comparison
    plt.subplot(2, 2, 4)
    r2_data = pd.DataFrame({
        'Original': [metrics['original_r2']],
        'Adjusted': [metrics['adjusted_r2']]
    }).melt()
    sns.barplot(data=r2_data, x='variable', y='value')
    plt.title('R-squared in Points Prediction')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()

def main():
    """Run the validation analysis."""
    # Load historical data
    pbp_data = load_historical_data()
    
    # Debug information
    print("\nColumns in play-by-play data:")
    print(sorted(pbp_data.columns))
    
    # Calculate original EDP
    original_edp = calculate_team_edp(pbp_data)
    
    print("\nColumns in original EDP:")
    print(sorted(original_edp.columns))
    
    original_edp = calculate_rolling_edp(original_edp)
    
    print("\nColumns in rolling EDP:")
    print(sorted(original_edp.columns))
    
    # Calculate opponent-adjusted EDP
    opponent_metrics = calculate_opponent_strength_metrics(
        original_edp,
        metrics=OPPONENT_STRENGTH_SETTINGS['metrics']
    )
    
    adjustments = calculate_matchup_adjustments(
        original_edp[['weighted_offensive_edp', 'weighted_defensive_edp']],
        opponent_metrics
    )
    
    adjusted_edp = original_edp.copy()
    adjusted_edp['adjusted_offensive_edp'] = apply_opponent_adjustments(
        original_edp,
        'weighted_offensive_edp',
        opponent_metrics,
        adjustments
    )
    adjusted_edp['adjusted_defensive_edp'] = apply_opponent_adjustments(
        original_edp,
        'weighted_defensive_edp',
        opponent_metrics,
        adjustments
    )
    
    # Calculate comparative metrics
    metrics = calculate_model_metrics(pbp_data, original_edp, adjusted_edp)
    
    # Print results
    print("\nModel Comparison Results:")
    print("-" * 50)
    print(f"Week-over-Week Stability:")
    print(f"  Original:  {metrics['original_wow_stability']:.3f}")
    print(f"  Adjusted:  {metrics['adjusted_wow_stability']:.3f}")
    print(f"\nPoints Prediction RMSE:")
    print(f"  Original:  {metrics['original_rmse']:.3f}")
    print(f"  Adjusted:  {metrics['adjusted_rmse']:.3f}")
    print(f"\nPoints Prediction R-squared:")
    print(f"  Original:  {metrics['original_r2']:.3f}")
    print(f"  Adjusted:  {metrics['adjusted_r2']:.3f}")
    
    # Create visualizations
    plot_model_comparison(original_edp, adjusted_edp, metrics)
    print("\nVisualization saved as 'model_comparison.png'")

if __name__ == "__main__":
    main() 