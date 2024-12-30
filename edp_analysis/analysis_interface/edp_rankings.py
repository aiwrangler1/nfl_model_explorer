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
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from edp_analysis.core_calculation.edp_calculator_v2 import EDPCalculator, EDPWeights
from edp_analysis.core_calculation.opponent_strength import OpponentStrengthAdjuster
from edp_analysis.data_management.data_loader import load_and_prepare_data
from edp_analysis.data_management.config import OUTPUT_CONFIG


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
    try:
        # Filter data for just this week
        week_mask = (df['game_id'].str.extract(r'2024_(\d+)')[0].fillna(0).astype(int) == week)
        df_week = df[week_mask].copy()
        
        if df_week.empty:
            print(f"No data found for Week {week}")
            return pd.DataFrame()
        
        # Filter out plays with null critical values
        critical_cols = ['posteam', 'defteam', 'drive', 'down', 'yardline_100']
        df_week = df_week.dropna(subset=critical_cols)
        
        # Standardize team names (LA -> LAR)
        df_week.loc[df_week['posteam'] == 'LA', 'posteam'] = 'LAR'
        df_week.loc[df_week['defteam'] == 'LA', 'defteam'] = 'LAR'
        
        # Preprocess data
        df_week = calculator.preprocess_data(df_week)
        
        # Calculate drive-level metrics
        drive_metrics = calculator.calculate_drive_quality(df_week)
        
        if drive_metrics.empty:
            print(f"No drive metrics calculated for Week {week}")
            return pd.DataFrame()
        
        # Get team metrics
        offensive_metrics, defensive_metrics = calculator.calculate_team_metrics(df_week)
        
        # Add season and week info before SoS adjustments
        offensive_metrics['season'] = df_week['season'].iloc[0]
        offensive_metrics['week'] = week
        defensive_metrics['season'] = df_week['season'].iloc[0]
        defensive_metrics['week'] = week
        
        # Add opponent information
        game_opponents = df_week.groupby('game_id').agg({
            'posteam': lambda x: sorted(x.unique())[0],  # Use sorted to ensure consistency
            'defteam': lambda x: sorted(x.unique())[0]
        })
        
        offensive_metrics['opponent'] = offensive_metrics['game_id'].map(
            lambda g: game_opponents.loc[g, 'defteam']
        )
        defensive_metrics['opponent'] = defensive_metrics['game_id'].map(
            lambda g: game_opponents.loc[g, 'posteam']
        )
        
        # Rename columns to match SoS adjuster expectations
        offensive_metrics = offensive_metrics.rename(columns={
            'earned_drive_points_off': 'offensive_edp',
            'drive_count_off': 'drive_count'
        })
        defensive_metrics = defensive_metrics.rename(columns={
            'earned_drive_points_def': 'defensive_edp',
            'drive_count_def': 'drive_count'
        })
        
        # Apply SoS adjustments
        sos_adjuster = OpponentStrengthAdjuster()
        adjusted_metrics = sos_adjuster.calculate_adjusted_metrics(offensive_metrics, defensive_metrics)
        
        # Create weekly EDP dataframe
        teams = sorted(list(adjusted_metrics['offensive'].keys()))  # Sort for consistency
        weekly_edp = pd.DataFrame({
            'team': teams,
            'earned_drive_points_off': [adjusted_metrics['offensive'][team] for team in teams],
            'earned_drive_points_def': [adjusted_metrics['defensive'][team] for team in teams],
            'drive_count_off': [offensive_metrics.set_index('team').get(('drive_count', team), 0) for team in teams],
            'drive_count_def': [defensive_metrics.set_index('team').get(('drive_count', team), 0) for team in teams],
            'game_id': [offensive_metrics.set_index('team').get(('game_id', team), '') for team in teams],
            'season': df_week['season'].iloc[0],
            'week': week
        })
        
        # Calculate per-drive metrics
        weekly_edp['offensive_edp_per_drive'] = (weekly_edp['earned_drive_points_off'] / 
                                                weekly_edp['drive_count_off'].clip(lower=1)).round(3)
        weekly_edp['defensive_edp_per_drive'] = (weekly_edp['earned_drive_points_def'] / 
                                                weekly_edp['drive_count_def'].clip(lower=1)).round(3)
        weekly_edp['total_edp_per_drive'] = (weekly_edp['offensive_edp_per_drive'] - 
                                            weekly_edp['defensive_edp_per_drive']).round(3)
        
        # Calculate total EDP
        weekly_edp['total_edp'] = (weekly_edp['earned_drive_points_off'] - 
                                  weekly_edp['earned_drive_points_def']).round(3)
        
        # Round metrics
        for col in ['earned_drive_points_off', 'earned_drive_points_def']:
            weekly_edp[col] = weekly_edp[col].round(3)
        
        return weekly_edp
        
    except Exception as e:
        print(f"Error processing Week {week}: {str(e)}")
        return pd.DataFrame()


def calculate_cumulative_edp(df: pd.DataFrame, max_week: int, calculator: EDPCalculator) -> pd.DataFrame:
    """Calculate cumulative EDP metrics through the specified week with recency weighting."""
    try:
        # Get weekly rankings first
        weekly_rankings = []
        for week in range(1, max_week + 1):
            print(f"\nProcessing Week {week}...")
            week_df = calculate_single_week_edp(df, week, calculator)
            if not week_df.empty:
                weekly_rankings.append(week_df)
        
        if not weekly_rankings:
            print("No valid weekly rankings calculated")
            return pd.DataFrame()
        
        print("\nCalculating season totals...")
        
        # Combine all weeks
        all_weeks = pd.concat(weekly_rankings, ignore_index=True)
        
        # Calculate recency weights
        max_week = all_weeks['week'].max()
        all_weeks['recency_weight'] = all_weeks['week'].apply(
            lambda w: 1 + (0.1 * (w - 1))  # 10% increase per week
        )
        
        # Calculate weighted season totals
        teams = sorted(all_weeks['team'].unique())  # Sort for consistency
        season_data = []
        
        for team in teams:
            team_data = all_weeks[all_weeks['team'] == team].copy()
            if not team_data.empty:
                weights = team_data['recency_weight'].values
                
                season_data.append({
                    'team': team,
                    'earned_drive_points_off': np.average(team_data['earned_drive_points_off'], weights=weights),
                    'earned_drive_points_def': np.average(team_data['earned_drive_points_def'], weights=weights),
                    'drive_count_off': team_data['drive_count_off'].sum(),
                    'drive_count_def': team_data['drive_count_def'].sum(),
                    'games_played': team_data['game_id'].nunique(),
                    'season': team_data['season'].iloc[0]
                })
        
        season_totals = pd.DataFrame(season_data)
        
        # Calculate per-drive metrics
        season_totals['offensive_edp_per_drive'] = (season_totals['earned_drive_points_off'] / 
                                                   season_totals['drive_count_off'].clip(lower=1)).round(3)
        season_totals['defensive_edp_per_drive'] = (season_totals['earned_drive_points_def'] / 
                                                   season_totals['drive_count_def'].clip(lower=1)).round(3)
        
        # Calculate total EDP metrics
        season_totals['total_edp'] = (season_totals['earned_drive_points_off'] - 
                                     season_totals['earned_drive_points_def']).round(3)
        season_totals['total_edp_per_drive'] = (season_totals['offensive_edp_per_drive'] - 
                                               season_totals['defensive_edp_per_drive']).round(3)
        
        # Round cumulative drive points
        season_totals['earned_drive_points_off'] = season_totals['earned_drive_points_off'].round(3)
        season_totals['earned_drive_points_def'] = season_totals['earned_drive_points_def'].round(3)
        
        # Reorder columns in a logical grouping
        column_order = [
            'team',
            'games_played',
            'offensive_edp_per_drive',
            'defensive_edp_per_drive',
            'total_edp_per_drive',
            'earned_drive_points_off',
            'earned_drive_points_def',
            'total_edp',
            'drive_count_off',
            'drive_count_def',
            'season'
        ]
        
        return season_totals[column_order]
        
    except Exception as e:
        print(f"Error calculating cumulative EDP: {str(e)}")
        return pd.DataFrame()


def validate_input_data(df: pd.DataFrame) -> None:
    """Validate input data has required columns and formats."""
    required_columns = {
        'game_id', 'play_id', 'posteam', 'defteam', 'week', 'season',
        'drive', 'down', 'yards_gained', 'play_type', 'yardline_100',
        'ydstogo', 'ep'  # Added ep for enhanced calculations
    }
    
    missing_cols = required_columns - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate game_id format
    invalid_game_ids = df[~df['game_id'].str.match(r'2024_\d+_\w+_\w+')]['game_id'].unique()
    if len(invalid_game_ids) > 0:
        raise ValueError(f"Invalid game_id format found: {invalid_game_ids}")
    
    # Check for null values in critical columns
    critical_columns = ['game_id', 'posteam', 'defteam', 'week', 'season', 'ep']
    null_counts = df[critical_columns].isnull().sum()
    if null_counts.any():
        print("\nWarning: Null values found in critical columns:")
        print(null_counts[null_counts > 0])


def process_season_data(df: pd.DataFrame, max_week: int, calculator: EDPCalculator) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Process all weeks up to max_week and return weekly and season rankings."""
    weekly_rankings = []
    
    print("\nWarning: Null values found in critical columns:")
    print(df[['posteam', 'defteam']].isnull().sum()[lambda x: x > 0])
    
    for week in range(1, max_week + 1):
        print(f"\nProcessing Week {week}...")
        week_df = calculate_single_week_edp(df, week, calculator)
        if not week_df.empty:
            weekly_rankings.append(week_df)
    
    print("\nCalculating season totals...")
    season_rankings = calculate_cumulative_edp(df, max_week, calculator)
    
    return weekly_rankings, season_rankings


def save_rankings_to_excel(
    weekly_rankings: List[pd.DataFrame],
    season_rankings: pd.DataFrame,
    max_week: int,
    output_dir: str = 'model_outputs'
) -> str:
    """Save weekly and season rankings to Excel."""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"edp_rankings_week{max_week}_{timestamp}.xlsx"
    filepath = os.path.join(output_dir, filename)
    
    # Create Excel writer
    with pd.ExcelWriter(filepath) as writer:
        # Save season rankings sorted by total_edp_per_drive
        season_rankings.sort_values('total_edp_per_drive', ascending=False).to_excel(
            writer,
            sheet_name='Season Rankings',
            index=False
        )
        
        # Save weekly rankings with game_id
        for i, week_df in enumerate(weekly_rankings, 1):
            # Add game_id to weekly rankings and reorder columns
            week_data = week_df.copy()
            
            # Reorder columns for weekly sheets
            weekly_cols = [
                'game_id',
                'team',
                'offensive_edp_per_drive',
                'defensive_edp_per_drive',
                'total_edp_per_drive',
                'earned_drive_points_off',
                'earned_drive_points_def',
                'total_edp',
                'drive_count_off',
                'drive_count_def',
                'week',
                'season'
            ]
            
            # Calculate weekly total metrics
            week_data['total_edp'] = (week_data['earned_drive_points_off'] - 
                                    week_data['earned_drive_points_def']).round(3)
            week_data['total_edp_per_drive'] = (
                (week_data['earned_drive_points_off'] / week_data['drive_count_off'] -
                 week_data['earned_drive_points_def'] / week_data['drive_count_def'])
            ).round(3)
            
            # Sort by game_id and team
            week_data = week_data.sort_values(['game_id', 'team'])
            
            # Select and order columns
            week_data = week_data[weekly_cols]
            
            week_data.to_excel(
                writer,
                sheet_name=f'Week {i}',
                index=False
            )
    
    print(f"\nRankings saved to: {filepath}")
    return filepath


def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Initialize calculator with default weights
    calculator = EDPCalculator()
    
    try:
        # Load data
        print("\nLoading play-by-play data...")
        df = load_and_prepare_data()
        
        # Determine max week if not specified
        if args.week is None:
            args.week = df['week'].max()
        
        print(f"\nAnalyzing through Week {args.week}...")
        
        # Process data and generate rankings
        weekly_rankings, season_rankings = process_season_data(df, args.week, calculator)
        
        # Save results
        save_rankings_to_excel(weekly_rankings, season_rankings, args.week)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 