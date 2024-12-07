"""
Validation script for the EDP-based game prediction model.

This script demonstrates the usage of the EDPGamePredictor class and evaluates
its performance on historical NFL data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from edp_analysis.game_prediction import EDPGamePredictor
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import os
import sys

# Add the parent directory to the path so we can import our other modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edp_analysis.edp_calculator import EDPCalculator
from edp_analysis.opponent_strength import OpponentStrengthAdjuster

def load_and_prepare_data() -> pd.DataFrame:
    """Load and prepare the NFL game data with EDP metrics."""
    # Initialize our EDP calculator and opponent strength adjuster
    calculator = EDPCalculator()
    adjuster = OpponentStrengthAdjuster()
    
    # Load raw play-by-play data and calculate base EDP
    df = calculator.load_and_process_data()
    print("Columns in raw data:", df.columns.tolist())
    
    # Calculate offensive and defensive EDP
    offensive_edp = calculator.calculate_offensive_edp(df)
    defensive_edp = calculator.calculate_defensive_edp(df)
    print("\nColumns in offensive EDP:", offensive_edp.columns.tolist())
    print("\nColumns in defensive EDP:", defensive_edp.columns.tolist())
    
    # Adjust for opponent strength
    adjusted_metrics = adjuster.calculate_adjusted_metrics(
        offensive_edp=offensive_edp,
        defensive_edp=defensive_edp
    )
    
    # Create adjusted offensive and defensive DataFrames
    adjusted_offensive = pd.DataFrame({
        'team': offensive_edp['team'],
        'game_id': offensive_edp['game_id'],
        'season': offensive_edp['season'],
        'week': offensive_edp['week'],
        'adjusted_offensive_edp': adjusted_metrics['offensive']
    })
    
    adjusted_defensive = pd.DataFrame({
        'team': defensive_edp['team'],
        'game_id': defensive_edp['game_id'],
        'season': defensive_edp['season'],
        'week': defensive_edp['week'],
        'adjusted_defensive_edp': adjusted_metrics['defensive']
    })
    
    # Merge adjusted metrics
    adjusted_df = pd.merge(
        adjusted_offensive,
        adjusted_defensive,
        on=['team', 'game_id', 'season', 'week'],
        how='outer'
    )
    
    # Merge game information
    games = df[['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score']].drop_duplicates()
    
    # Create a DataFrame with one row per team per game
    game_metrics = []
    
    for _, game in games.iterrows():
        # Get adjusted metrics for this game
        game_metrics_df = adjusted_df[adjusted_df['game_id'] == game['game_id']]
        
        # Home team metrics
        home_metrics_df = game_metrics_df[game_metrics_df['team'] == game['home_team']]
        if len(home_metrics_df) > 0:
            home_metrics = {
                'game_id': game['game_id'],
                'season': game['season'],
                'week': game['week'],
                'team': game['home_team'],
                'opponent': game['away_team'],
                'home_away': 'home',
                'points': game['home_score'],
                'opponent_points': game['away_score'],
                'adjusted_offensive_edp': home_metrics_df['adjusted_offensive_edp'].iloc[0],
                'adjusted_defensive_edp': home_metrics_df['adjusted_defensive_edp'].iloc[0]
            }
            game_metrics.append(home_metrics)
        
        # Away team metrics
        away_metrics_df = game_metrics_df[game_metrics_df['team'] == game['away_team']]
        if len(away_metrics_df) > 0:
            away_metrics = {
                'game_id': game['game_id'],
                'season': game['season'],
                'week': game['week'],
                'team': game['away_team'],
                'opponent': game['home_team'],
                'home_away': 'away',
                'points': game['away_score'],
                'opponent_points': game['home_score'],
                'adjusted_offensive_edp': away_metrics_df['adjusted_offensive_edp'].iloc[0],
                'adjusted_defensive_edp': away_metrics_df['adjusted_defensive_edp'].iloc[0]
            }
            game_metrics.append(away_metrics)
    
    df = pd.DataFrame(game_metrics)
    print("\nColumns in final data:", df.columns.tolist())
    
    # Convert season and week to numeric
    df['season'] = pd.to_numeric(df['season'])
    df['week'] = pd.to_numeric(df['week'])
    
    # Sort by season and week
    df = df.sort_values(['season', 'week'])
    
    return df

def create_temporal_splits(
    df: pd.DataFrame,
    n_splits: int = 3
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Create temporal train-test splits for model validation.
    
    Args:
        df: DataFrame with game data
        n_splits: Number of temporal splits to create
        
    Returns:
        List of (train, test) DataFrame pairs
    """
    splits = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    # Sort by season and week
    df = df.sort_values(['season', 'week'])
    
    for train_idx, test_idx in tscv.split(df):
        train = df.iloc[train_idx]
        test = df.iloc[test_idx]
        splits.append((train, test))
        
    return splits

def evaluate_model_performance(
    predictor: EDPGamePredictor,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray
) -> Dict[str, float]:
    """
    Fit and evaluate the model on a train-test split.
    
    Args:
        predictor: EDPGamePredictor instance
        X_train: Training features
        y_train: Training targets
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Fit the model
    predictor.fit_hierarchical_model(X_train, y_train)
    
    # Get predictions and metrics
    metrics = predictor.evaluate_calibration(X_test, y_test)
    
    return metrics

def plot_calibration_results(
    predictor: EDPGamePredictor,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    save_path: str = None
):
    """
    Create calibration plots for model predictions.
    
    Args:
        predictor: Fitted EDPGamePredictor instance
        X_test: Test features
        y_test: Test targets
        save_path: Optional path to save the plot
    """
    predictions, lower, upper = predictor.predict(X_test)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Predicted vs Actual
    ax1.scatter(predictions, y_test, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel('Predicted Point Differential')
    ax1.set_ylabel('Actual Point Differential')
    ax1.set_title('Predicted vs Actual Point Differentials')
    
    # Plot 2: Prediction Intervals
    sorted_idx = np.argsort(predictions)
    ax2.fill_between(
        range(len(predictions)),
        lower[sorted_idx],
        upper[sorted_idx],
        alpha=0.3,
        label='95% Prediction Interval'
    )
    ax2.plot(range(len(predictions)), predictions[sorted_idx], 'b-', label='Predicted')
    ax2.plot(range(len(predictions)), y_test[sorted_idx], 'r.', alpha=0.5, label='Actual')
    ax2.set_xlabel('Sorted Game Index')
    ax2.set_ylabel('Point Differential')
    ax2.set_title('Prediction Intervals')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()

def main():
    """Main validation routine."""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("\nCreating predictor instance...")
    predictor = EDPGamePredictor(
        model_type="hierarchical_bayes",
        mcmc_samples=2000
    )
    
    print("\nPreparing features...")
    feature_df = predictor.prepare_matchup_features(df)
    
    print("\nFeature columns:", feature_df.columns.tolist())
    print(f"Number of games: {len(feature_df)}")
    
    # Create temporal splits
    splits = create_temporal_splits(feature_df)
    
    # Evaluate on each split
    all_metrics = []
    for i, (train, test) in enumerate(splits):
        print(f"\nEvaluating split {i+1}/{len(splits)}")
        
        X_train = train.drop('point_differential', axis=1)
        y_train = train['point_differential'].values
        X_test = test.drop('point_differential', axis=1)
        y_test = test['point_differential'].values
        
        metrics = evaluate_model_performance(
            predictor,
            X_train,
            y_train,
            X_test,
            y_test
        )
        
        print("\nMetrics for this split:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.3f}")
            
        all_metrics.append(metrics)
        
        # Create calibration plots
        plot_calibration_results(
            predictor,
            X_test,
            y_test,
            save_path=f"model_outputs/calibration_plot_split_{i+1}.png"
        )
    
    # Print average metrics across splits
    print("\nAverage metrics across all splits:")
    avg_metrics = {}
    for metric in all_metrics[0].keys():
        avg_metrics[metric] = np.mean([m[metric] for m in all_metrics])
        print(f"{metric}: {avg_metrics[metric]:.3f}")

if __name__ == "__main__":
    main() 