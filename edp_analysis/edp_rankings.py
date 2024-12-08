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
    team_edp = calculator.calculate_team_edp(df_week)
    
    # Clean up columns - remove duplicate season/week columns
    team_edp = team_edp.loc[:, ~team_edp.columns.str.endswith(('_x', '_y'))]
    
    # Calculate total EDP
    team_edp['total_edp'] = team_edp['offensive_edp'] + team_edp['defensive_edp']
    
    # Add points and drive data
    points_data = df_week.groupby(['posteam', 'game_id']).agg({
        'points': 'sum',
        'drive': 'nunique'  # Count unique drives
    }).reset_index()
    points_against = df_week.groupby(['defteam', 'game_id']).agg({
        'points': 'sum',
        'drive': 'nunique'  # Count unique drives
    }).reset_index()
    
    # Merge points and drive data
    team_edp = pd.merge(
        team_edp,
        points_data.rename(columns={
            'posteam': 'team', 
            'points': 'points_scored',
            'drive': 'offensive_drives'
        }),
        on=['team', 'game_id'],
        how='left'
    )
    team_edp = pd.merge(
        team_edp,
        points_against.rename(columns={
            'defteam': 'team', 
            'points': 'points_allowed',
            'drive': 'defensive_drives'
        }),
        on=['team', 'game_id'],
        how='left'
    )
    
    # Calculate per-drive metrics
    team_edp['off_edp_per_drive'] = (team_edp['offensive_edp'] / 
                                    team_edp['offensive_drives']).round(3)
    team_edp['def_edp_per_drive'] = (team_edp['defensive_edp'] / 
                                    team_edp['defensive_drives']).round(3)
    team_edp['total_edp_per_drive'] = (team_edp['off_edp_per_drive'] + 
                                      team_edp['def_edp_per_drive']).round(3)
    
    # Round EDP metrics
    edp_cols = ['offensive_edp', 'defensive_edp', 'total_edp']
    team_edp[edp_cols] = team_edp[edp_cols].round(3)
    
    # Select and order columns
    columns = [
        'team', 'game_id',
        'total_edp', 'offensive_edp', 'defensive_edp',
        'total_edp_per_drive', 'off_edp_per_drive', 'def_edp_per_drive',
        'points_scored', 'points_allowed',
        'offensive_drives', 'defensive_drives'
    ]
    
    return team_edp[columns]


def calculate_cumulative_edp(df: pd.DataFrame, max_week: int, calculator: EDPCalculator) -> pd.DataFrame:
    """
    Calculate cumulative SoS-adjusted EDP metrics through the specified week.
    
    Args:
        df: Play-by-play data
        max_week: Latest week to include
        calculator: EDPCalculator instance
        
    Returns:
        DataFrame with season-total team metrics
    """
    # Filter data through max week
    week_mask = (df['game_id']
                .str.extract(r'2024_(\d+)')[0]
                .fillna(0)
                .astype(int) <= max_week)
    df_through_week = df[week_mask]
    
    # Use EDPCalculator to prepare data and calculate metrics
    df_through_week = calculator.prepare_play_data(df_through_week)
    
    # Get raw team EDP metrics
    team_edp = calculator.calculate_team_edp(df_through_week)
    
    # Add season and week information from game_id
    team_edp['season'] = 2024
    team_edp['week'] = team_edp['game_id'].str.extract(r'2024_(\d+)')[0].astype(int)
    
    # Create OpponentStrengthAdjuster and calculate adjusted metrics
    strength_adjuster = OpponentStrengthAdjuster()
    adjusted_metrics = strength_adjuster.calculate_adjusted_metrics(
        offensive_edp=team_edp[['team', 'game_id', 'season', 'week', 'offensive_edp']],
        defensive_edp=team_edp[['team', 'game_id', 'season', 'week', 'defensive_edp']]
    )
    
    # Convert adjusted metrics to DataFrame
    adjusted_offensive = pd.Series(adjusted_metrics['offensive'])
    adjusted_defensive = pd.Series(adjusted_metrics['defensive'])
    
    # Add points and drive data
    points_data = df_through_week.groupby('posteam').agg({
        'points': 'sum',
        'drive': 'nunique'  # Count unique drives
    }).reset_index()
    points_against = df_through_week.groupby('defteam').agg({
        'points': 'sum',
        'drive': 'nunique'  # Count unique drives
    }).reset_index()
    
    # Create season totals DataFrame with SoS-adjusted metrics
    season_totals = pd.DataFrame({
        'team': adjusted_offensive.index,
        'offensive_edp': adjusted_offensive.values,
        'defensive_edp': adjusted_defensive.values
    })
    
    # Merge points and drive data
    season_totals = pd.merge(
        season_totals,
        points_data.rename(columns={
            'posteam': 'team', 
            'points': 'points_scored',
            'drive': 'offensive_drives'
        }),
        on='team',
        how='left'
    )
    season_totals = pd.merge(
        season_totals,
        points_against.rename(columns={
            'defteam': 'team', 
            'points': 'points_allowed',
            'drive': 'defensive_drives'
        }),
        on='team',
        how='left'
    )
    
    # Calculate total EDP and point differential
    season_totals['total_edp'] = (season_totals['offensive_edp'] + 
                                 season_totals['defensive_edp']).round(3)
    season_totals['point_differential'] = (season_totals['points_scored'] - 
                                         season_totals['points_allowed'])
    
    # Calculate per-drive metrics
    season_totals['off_edp_per_drive'] = (season_totals['offensive_edp'] / 
                                         season_totals['offensive_drives']).round(3)
    season_totals['def_edp_per_drive'] = (season_totals['defensive_edp'] / 
                                         season_totals['defensive_drives']).round(3)
    season_totals['total_edp_per_drive'] = (season_totals['off_edp_per_drive'] + 
                                           season_totals['def_edp_per_drive']).round(3)
    
    # Calculate games played
    games_played = team_edp.groupby('team').size()
    season_totals['games_played'] = season_totals['team'].map(games_played)
    
    # Round EDP metrics
    edp_cols = ['offensive_edp', 'defensive_edp', 'total_edp']
    season_totals[edp_cols] = season_totals[edp_cols].round(3)
    
    # Sort by total EDP
    season_totals = season_totals.sort_values('total_edp', ascending=False)
    
    # Select and order columns
    columns = [
        'team',
        'total_edp', 'offensive_edp', 'defensive_edp',
        'total_edp_per_drive', 'off_edp_per_drive', 'def_edp_per_drive',
        'point_differential', 'points_scored', 'points_allowed',
        'offensive_drives', 'defensive_drives',
        'games_played'
    ]
    
    return season_totals[columns]


def process_season_data(df: pd.DataFrame, max_week: int, calculator: EDPCalculator) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Process EDP data for all weeks and calculate cumulative stats."""
    weekly_rankings = []
    
    # Calculate individual week stats
    for week in range(1, max_week + 1):
        print(f"\nProcessing Week {week}...")
        week_rankings = calculate_single_week_edp(df, week, calculator)
        if week_rankings is not None:
            weekly_rankings.append(week_rankings)
    
    # Calculate cumulative stats
    print("\nCalculating season totals...")
    cumulative_rankings = calculate_cumulative_edp(df, max_week, calculator)
    
    return weekly_rankings, cumulative_rankings


def save_rankings(weekly_rankings: List[pd.DataFrame],
                 cumulative_rankings: pd.DataFrame,
                 max_week: int) -> str:
    """Save weekly and cumulative rankings to Excel."""
    # Create output folder
    output_folder = os.path.join(project_root, "model_outputs", "rankings", f"week_{max_week}")
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%b %d-%I:%M%p").lower()
    excel_filename = os.path.join(
        output_folder,
        f"nfl_edp_rankings_2024_week_{max_week}_{timestamp}.xlsx"
    )
    
    # Save to Excel with a sheet for each week plus cumulative
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Save each individual week
        for week, week_data in enumerate(weekly_rankings, 1):
            # Sort by total EDP (offensive + defensive)
            week_data['total_edp'] = week_data['offensive_edp'] + week_data['defensive_edp']
            week_data = week_data.sort_values('total_edp', ascending=False)
            
            # Round numeric columns
            numeric_cols = week_data.select_dtypes(include=['float64']).columns
            week_data[numeric_cols] = week_data[numeric_cols].round(3)
            
            week_data.to_excel(writer, sheet_name=f'Week_{week}', index=False)
        
        # Save cumulative stats (already properly formatted from calculate_cumulative_edp)
        numeric_cols = cumulative_rankings.select_dtypes(include=['float64']).columns
        cumulative_rankings[numeric_cols] = cumulative_rankings[numeric_cols].round(3)
        cumulative_rankings.to_excel(writer, sheet_name='Season_Total', index=False)
    
    return excel_filename


def main() -> None:
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    try:
        # Initialize EDP calculator
        calculator = EDPCalculator()
        
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        years = [2024]
        df_all_years = load_and_prepare_data(years)
        
        # Determine latest week if not specified
        if args.week:
            max_week = args.week
        else:
            max_week = (df_all_years['game_id']
                       .str.extract(r'2024_(\d+)')[0]
                       .fillna(0)
                       .astype(int)
                       .max())
        
        print(f"\nAnalyzing data through Week {max_week}")
        
        # Process all weeks and calculate cumulative stats
        weekly_rankings, cumulative_rankings = process_season_data(
            df_all_years, max_week, calculator
        )
        
        # Save results
        output_file = save_rankings(weekly_rankings, cumulative_rankings, max_week)
        
        print(f"\nRankings saved to: {output_file}")
        print("\nTop 5 teams by Total EDP:")
        print(cumulative_rankings[['team', 'total_edp', 'offensive_edp', 
                                 'defensive_edp', 'games_played']].head())
        
    except Exception as e:
        print(f"\nError generating rankings: {str(e)}")
        raise


if __name__ == "__main__":
    main() 