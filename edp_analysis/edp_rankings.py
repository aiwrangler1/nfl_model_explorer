#!/usr/bin/env python3
"""
NFL EDP Rankings Generator

This script generates both weekly and cumulative Expected Drive Points (EDP) rankings
for NFL teams, with strength of schedule adjustments.

Usage:
    python edp_rankings.py [--week WEEK]
    
    --week: Optional. Specify which week to analyze through (1-18).
            If not provided, will use the latest week with available data.
"""

import sys
import os
import datetime
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from edp_analysis.edp_calculator import EDPCalculator
from edp_analysis.data_loader import load_and_prepare_data
from edp_analysis.opponent_strength import OpponentStrengthAdjuster


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate NFL EDP rankings with SoS adjustments."
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Week to analyze through (1-18). If not provided, uses latest available week.",
        choices=range(1, 19),
        metavar="WEEK"
    )
    return parser.parse_args()


def calculate_single_week_edp(df: pd.DataFrame, week: int, calculator: EDPCalculator) -> pd.DataFrame:
    """Calculate EDP metrics for a single week."""
    # Filter data for just this week
    week_mask = (df['game_id']
                .str.extract(r'2024_(\d+)')[0]
                .fillna(0)
                .astype(int) == week)
    df_week = df[week_mask].copy()
    
    # Use EDPCalculator to prepare data and calculate metrics
    df_week = calculator.prepare_play_data(df_week)
    
    # Calculate team EDP
    team_edp = calculator.calculate_team_edp(df_week)
    
    # Store raw metrics before any adjustments
    raw_metrics = team_edp.copy()
    raw_metrics = raw_metrics.rename(columns={
        'offensive_edp': 'raw_offensive_edp',
        'defensive_edp': 'raw_defensive_edp',
        'offensive_edp_per_drive': 'raw_offensive_edp_per_drive',
        'defensive_edp_per_drive': 'raw_defensive_edp_per_drive',
        'total_edp': 'raw_total_edp',
        'total_edp_per_drive': 'raw_total_edp_per_drive'
    })
    
    # Get final game scores using the score_differential column
    game_scores = df_week.groupby('game_id').agg({
        'posteam': 'first',
        'defteam': 'first',
        'score_differential': 'last',  # Get final score differential
        'total_home_score': 'last',    # Get final home score
        'total_away_score': 'last'     # Get final away score
    }).reset_index()
    
    # Create separate records for home and away teams
    home_scores = game_scores.copy()
    home_scores['team'] = home_scores['posteam']
    home_scores['points_scored'] = home_scores['total_home_score']
    home_scores['points_allowed'] = home_scores['total_away_score']
    
    away_scores = game_scores.copy()
    away_scores['team'] = away_scores['defteam']
    away_scores['points_scored'] = away_scores['total_away_score']
    away_scores['points_allowed'] = away_scores['total_home_score']
    
    game_scores_final = pd.concat([home_scores, away_scores])
    
    # Merge final scores into raw_metrics
    raw_metrics = pd.merge(
        raw_metrics,
        game_scores_final[['game_id', 'team', 'points_scored', 'points_allowed']],
        on=['team', 'game_id'],
        how='left'
    )
    
    # Add point differential
    raw_metrics['point_differential'] = raw_metrics['points_scored'] - raw_metrics['points_allowed']
    
    # Ensure all required columns are present
    required_columns = [
        'team', 'game_id',
        'raw_total_edp', 'raw_offensive_edp', 'raw_defensive_edp',
        'raw_total_edp_per_drive', 'raw_offensive_edp_per_drive', 'raw_defensive_edp_per_drive',
        'points_scored', 'points_allowed', 'point_differential',
        'offensive_drives', 'defensive_drives'
    ]
    
    missing_cols = [col for col in required_columns if col not in raw_metrics.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    return raw_metrics[required_columns]


def calculate_cumulative_edp(df: pd.DataFrame, max_week: int, calculator: EDPCalculator) -> pd.DataFrame:
    """
    Calculate cumulative EDP metrics through the specified week.
    """
    # Filter data through max week
    week_mask = (df['game_id']
                .str.extract(r'2024_(\d+)')[0]
                .fillna(0)
                .astype(int) <= max_week)
    df_through_week = df[week_mask]
    
    # Use EDPCalculator to prepare data and calculate metrics
    df_through_week = calculator.prepare_play_data(df_through_week)
    
    # Calculate team EDP for each game
    game_edp = calculator.calculate_team_edp(df_through_week)
    
    # Calculate games played
    games_played = game_edp.groupby('team').size().reset_index(name='games_played')
    
    # Calculate raw season totals by summing up game values
    raw_season_totals = game_edp.groupby('team').agg({
        'offensive_edp': 'sum',
        'defensive_edp': 'sum',
        'offensive_drives': 'sum',
        'defensive_drives': 'sum'
    }).reset_index()
    
    # Calculate raw total EDP
    raw_season_totals['raw_offensive_edp'] = raw_season_totals['offensive_edp']
    raw_season_totals['raw_defensive_edp'] = raw_season_totals['defensive_edp']
    raw_season_totals['raw_total_edp'] = raw_season_totals['raw_offensive_edp'] + raw_season_totals['raw_defensive_edp']
    
    # Calculate raw per-drive metrics
    raw_season_totals['raw_offensive_edp_per_drive'] = raw_season_totals['raw_offensive_edp'] / raw_season_totals['offensive_drives']
    raw_season_totals['raw_defensive_edp_per_drive'] = raw_season_totals['raw_defensive_edp'] / raw_season_totals['defensive_drives']
    raw_season_totals['raw_total_edp_per_drive'] = raw_season_totals['raw_total_edp'] / raw_season_totals['offensive_drives']
    
    # Calculate SoS adjustments
    strength_adjuster = OpponentStrengthAdjuster()
    sos_game_adjustments = strength_adjuster.calculate_adjusted_metrics(
        offensive_edp=game_edp[['team', 'game_id', 'season', 'week', 'offensive_edp']],
        defensive_edp=game_edp[['team', 'game_id', 'season', 'week', 'defensive_edp']]
    )
    
    # Create game-level SoS adjustments DataFrame
    sos_factors = pd.DataFrame({
        'team': list(sos_game_adjustments['offensive'].keys()),
        'sos_offensive_factor': [v for v in sos_game_adjustments['offensive'].values()],
        'sos_defensive_factor': [v for v in sos_game_adjustments['defensive'].values()]
    })
    
    # Normalize SoS factors to center around 1.0
    sos_factors['sos_offensive_factor'] = sos_factors['sos_offensive_factor'] / sos_factors['sos_offensive_factor'].mean()
    sos_factors['sos_defensive_factor'] = sos_factors['sos_defensive_factor'] / sos_factors['sos_defensive_factor'].mean()
    
    # Apply SoS adjustments to raw season totals
    season_totals = pd.merge(raw_season_totals, sos_factors, on='team', how='left')
    season_totals['sos_offensive_edp'] = season_totals['raw_offensive_edp'] * season_totals['sos_offensive_factor']
    season_totals['sos_defensive_edp'] = season_totals['raw_defensive_edp'] * season_totals['sos_defensive_factor']
    season_totals['sos_total_edp'] = season_totals['sos_offensive_edp'] + season_totals['sos_defensive_edp']
    
    # Calculate SoS per-drive metrics
    season_totals['sos_offensive_edp_per_drive'] = season_totals['sos_offensive_edp'] / season_totals['offensive_drives']
    season_totals['sos_defensive_edp_per_drive'] = season_totals['sos_defensive_edp'] / season_totals['defensive_drives']
    season_totals['sos_total_edp_per_drive'] = season_totals['sos_total_edp'] / season_totals['offensive_drives']
    
    # Add games played
    season_totals = pd.merge(season_totals, games_played, on='team', how='left')
    
    # Sort by total EDP
    season_totals = season_totals.sort_values('raw_total_edp', ascending=False)
    
    return season_totals


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate input data has required columns and formats."""
    required_columns = {
        'game_id', 'play_id', 'posteam', 'defteam', 'week', 'season',
        'drive', 'down', 'yards_gained', 'play_type', 'yardline_100',
        'ydstogo', 'total_home_score', 'total_away_score'
    }
    
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate game_id format
    invalid_game_ids = df[~df['game_id'].str.match(r'2024_\d+_\w+_\w+')]['game_id'].unique()
    if len(invalid_game_ids) > 0:
        raise ValueError(f"Invalid game_id format found: {invalid_game_ids}")
    
    # Check for null values in critical columns
    critical_columns = ['game_id', 'posteam', 'defteam', 'week', 'season']
    null_counts = df[critical_columns].isnull().sum()
    if null_counts.any():
        print("\nWarning: Null values found in critical columns:")
        print(null_counts[null_counts > 0])


def process_season_data(df: pd.DataFrame, max_week: int, calculator: EDPCalculator) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Process EDP data for all weeks and calculate cumulative stats."""
    try:
        # Validate input data
        validate_input_data(df)
        
        weekly_rankings = []
        
        # Calculate individual week stats
        for week in range(1, max_week + 1):
            print(f"\nProcessing Week {week}...")
            try:
                week_rankings = calculate_single_week_edp(df, week, calculator)
                if week_rankings is not None and not week_rankings.empty:
                    weekly_rankings.append(week_rankings)
                else:
                    print(f"Warning: No data found for Week {week}")
            except Exception as e:
                print(f"Error processing Week {week}: {str(e)}")
                continue
        
        if not weekly_rankings:
            raise ValueError("No weekly rankings could be calculated")
        
        # Calculate cumulative stats
        print("\nCalculating season totals...")
        cumulative_rankings = calculate_cumulative_edp(df, max_week, calculator)
        
        # Validate output
        if cumulative_rankings.empty:
            raise ValueError("Cumulative rankings calculation failed")
            
        return weekly_rankings, cumulative_rankings
        
    except Exception as e:
        print(f"\nError processing season data: {str(e)}")
        raise


def save_rankings_to_excel(
    weekly_rankings: List[pd.DataFrame],
    season_rankings: pd.DataFrame,
    max_week: int,
    output_dir: str = 'model_outputs'
) -> str:
    """
    Save weekly and season rankings to Excel.
    
    Args:
        weekly_rankings: List of weekly ranking DataFrames
        season_rankings: Season-to-date rankings DataFrame
        max_week: Latest week included in the rankings
        output_dir: Directory to save the output file
        
    Returns:
        str: Path to the saved Excel file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp and week
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    excel_filename = os.path.join(
        output_dir,
        f'edp_rankings_week{max_week}_{timestamp}.xlsx'
    )
    
    # Define column sets
    season_columns = [
        'team', 'games_played',
        # Raw metrics
        'raw_total_edp', 'raw_offensive_edp', 'raw_defensive_edp',
        'raw_total_edp_per_drive', 'raw_offensive_edp_per_drive', 'raw_defensive_edp_per_drive',
        # SoS-adjusted metrics
        'sos_total_edp', 'sos_offensive_edp', 'sos_defensive_edp',
        'sos_total_edp_per_drive', 'sos_offensive_edp_per_drive', 'sos_defensive_edp_per_drive',
        # Recency-weighted metrics
        'weighted_total_edp', 'weighted_offensive_edp', 'weighted_defensive_edp',
        'weighted_total_edp_per_drive', 'weighted_offensive_edp_per_drive', 'weighted_defensive_edp_per_drive',
        # Blended metrics (SoS + Recency)
        'blended_total_edp', 'blended_offensive_edp', 'blended_defensive_edp',
        'blended_total_edp_per_drive', 'blended_offensive_edp_per_drive', 'blended_defensive_edp_per_drive',
        # Additional stats
        'point_differential', 'points_scored', 'points_allowed',
        'points_per_game', 'points_allowed_per_game',
        'offensive_drives', 'defensive_drives'
    ]
    
    weekly_columns = [
        'team', 'game_id',
        # Raw metrics for the week
        'raw_total_edp', 'raw_offensive_edp', 'raw_defensive_edp',
        'raw_total_edp_per_drive', 'raw_offensive_edp_per_drive', 'raw_defensive_edp_per_drive',
        # Game stats
        'points_scored', 'points_allowed', 'point_differential',
        'offensive_drives', 'defensive_drives'
    ]
    
    # Filter columns to only those that exist in the DataFrames
    season_columns = [col for col in season_columns if col in season_rankings.columns]
    
    # Create Excel writer
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Write season-to-date rankings
        season_rankings.to_excel(
            writer,
            sheet_name='season_total',
            index=False,
            columns=season_columns
        )
        
        # Write weekly rankings
        for week_num, week_df in enumerate(weekly_rankings, 1):
            # Filter columns to only those that exist in this week's DataFrame
            available_columns = [col for col in weekly_columns if col in week_df.columns]
            
            week_df.to_excel(
                writer,
                sheet_name=f'week_{week_num}',
                index=False,
                columns=available_columns
            )
    
    return excel_filename


def calculate_season_rankings(weekly_rankings: List[pd.DataFrame]) -> pd.DataFrame:
    """Calculate cumulative season-to-date rankings."""
    if not weekly_rankings:
        return pd.DataFrame()
        
    # Concatenate all weekly rankings
    all_weeks = pd.concat(weekly_rankings, ignore_index=True)
    
    # Calculate season totals
    season_totals = all_weeks.groupby('team').agg({
        'offensive_edp': 'sum',
        'defensive_edp': 'sum',
        'total_edp': 'sum',
        'points_scored': 'sum',
        'points_allowed': 'sum',
        'offensive_drives': 'sum',
        'defensive_drives': 'sum',
        'game_id': 'count'  # Number of games played
    }).reset_index()
    
    # Calculate per-drive and per-game metrics
    season_totals['offensive_edp_per_drive'] = (
        season_totals['offensive_edp'] / season_totals['offensive_drives']
    )
    season_totals['defensive_edp_per_drive'] = (
        season_totals['defensive_edp'] / season_totals['defensive_drives']
    )
    season_totals['total_edp_per_drive'] = (
        season_totals['total_edp'] / season_totals['offensive_drives']
    )
    
    # Add point differential
    season_totals['point_differential'] = (
        season_totals['points_scored'] - season_totals['points_allowed']
    )
    
    # Add per-game metrics
    season_totals['points_per_game'] = (
        season_totals['points_scored'] / season_totals['game_id']
    )
    season_totals['points_allowed_per_game'] = (
        season_totals['points_allowed'] / season_totals['game_id']
    )
    
    # Sort by total EDP
    season_totals = season_totals.sort_values('total_edp', ascending=False)
    
    # Round numeric columns
    numeric_cols = season_totals.select_dtypes(include=[np.number]).columns
    season_totals[numeric_cols] = season_totals[numeric_cols].round(3)
    
    return season_totals


def main() -> None:
    """Main execution function."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        print("\nLoading and preprocessing data...")
        df = load_and_prepare_data()
        
        if df.empty:
            raise ValueError("No data loaded")
            
        # Determine max week
        max_week = args.week
        if max_week is None:
            max_week = df['game_id'].str.extract(r'2024_(\d+)')[0].astype(int).max()
            if pd.isna(max_week):
                raise ValueError("Could not determine maximum week from data")
        
        print(f"\nAnalyzing data through Week {max_week}")
        
        # Initialize calculator
        calculator = EDPCalculator()
        
        # Process data and generate rankings
        weekly_rankings, cumulative_rankings = process_season_data(
            df, max_week, calculator
        )
        
        # Validate rankings before saving
        if not weekly_rankings or cumulative_rankings.empty:
            raise ValueError("No rankings generated")
            
        # Save results
        output_file = save_rankings_to_excel(
            weekly_rankings,
            cumulative_rankings,
            max_week
        )
        
        print(f"\nRankings saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError generating rankings: {str(e)}")
        raise


if __name__ == "__main__":
    main() 