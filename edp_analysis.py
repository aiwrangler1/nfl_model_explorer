"""Main script for NFL EDP analysis."""

import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

from data_loader import load_and_prepare_data
from edp_calculator import EDPCalculator

def apply_recency_weights(df: pd.DataFrame, current_week: int) -> pd.DataFrame:
    """Apply recency-based weights to weekly metrics.
    
    Weights:
    - Last 4 games: 1.0
    - After that: Exponential decay with medium strength (decay factor ~0.85)
    
    Note: Properly handles bye weeks by counting actual games played.
    """
    df = df.copy()
    
    # Sort by team and week to properly count games played
    df = df.sort_values(['team', 'week'])
    
    # For each team, calculate the number of actual games back
    def calculate_games_back(group):
        group = group.copy()
        # Count actual games, not just weeks
        actual_games = range(len(group)-1, -1, -1)
        group['games_back'] = actual_games
        return group
    
    df = df.groupby('team', group_keys=False).apply(calculate_games_back)
    
    # Apply weights
    # 1.0 for last 4 games, then exponential decay
    decay_factor = 0.85  # Medium strength decay (0.85^5 ≈ 0.44 after 5 more games)
    
    def calculate_weight(games_back):
        if games_back < 4:
            return 1.0
        else:
            return decay_factor ** (games_back - 3)  # Start decay after 4th game
    
    df['weight'] = df['games_back'].apply(calculate_weight)
    
    return df

def analyze_week(df: pd.DataFrame, week: int) -> pd.DataFrame:
    """Analyze a single week of data."""
    # Filter for week
    df_week = df[df['week'] == week].copy()
    if df_week.empty:
        print(f"No data for Week {week}")
        return pd.DataFrame()
    
    # Calculate metrics
    calculator = EDPCalculator()
    drive_metrics = calculator.calculate_drive_metrics(df_week)
    offensive, defensive = calculator.calculate_team_metrics(drive_metrics)
    
    # Merge offensive and defensive metrics
    weekly_metrics = pd.merge(
        offensive,
        defensive,
        on=['team', 'game_id', 'week'],
        how='outer'
    )
    
    # Calculate per-drive metrics
    weekly_metrics['offensive_edp_per_drive'] = (
        weekly_metrics['earned_drive_points_off'] / 
        weekly_metrics['drive_count_off']
    ).round(3)
    
    weekly_metrics['defensive_edp_per_drive'] = (
        weekly_metrics['earned_drive_points_def'] / 
        weekly_metrics['drive_count_def']
    ).round(3)
    
    weekly_metrics['total_edp_per_drive'] = (
        weekly_metrics['offensive_edp_per_drive'] - 
        weekly_metrics['defensive_edp_per_drive']
    ).round(3)
    
    return weekly_metrics

def round_appropriately(x):
    """Round based on value magnitude:
    - Numbers >= 100: 1 decimal point
    - Numbers < 100: 3 decimal points
    """
    if abs(x) >= 100:
        return round(x, 1)
    return round(x, 3)

def calculate_season_metrics(df: pd.DataFrame, current_week: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate both weekly and season-to-date metrics with opponent adjustments."""
    all_weekly_metrics = pd.DataFrame()
    calculator = EDPCalculator()
    
    # Calculate metrics for each week
    for week in range(1, current_week + 1):
        weekly = analyze_week(df, week)
        if not weekly.empty:
            all_weekly_metrics = pd.concat([all_weekly_metrics, weekly])
    
    # Apply recency weights
    weighted_metrics = apply_recency_weights(all_weekly_metrics, current_week)
    
    # Calculate raw metrics
    raw_metrics = all_weekly_metrics.groupby('team').agg({
        'earned_drive_points_off': 'sum',
        'earned_drive_points_def': 'sum',
        'drive_count_off': 'sum',
        'drive_count_def': 'sum'
    }).reset_index()
    
    # Calculate raw total metrics first
    raw_metrics['raw_off_edp'] = raw_metrics['earned_drive_points_off'].apply(round_appropriately)
    raw_metrics['raw_def_edp'] = raw_metrics['earned_drive_points_def'].apply(round_appropriately)
    raw_metrics['raw_total_edp'] = (
        raw_metrics['raw_off_edp'] - 
        raw_metrics['raw_def_edp']
    ).apply(round_appropriately)
    
    # Then calculate raw per-drive metrics
    raw_metrics['raw_off_edp_per_drive'] = (
        raw_metrics['raw_off_edp'] / 
        raw_metrics['drive_count_off']
    ).round(3)
    raw_metrics['raw_def_edp_per_drive'] = (
        raw_metrics['raw_def_edp'] / 
        raw_metrics['drive_count_def']
    ).round(3)
    raw_metrics['raw_edp_per_drive'] = (
        raw_metrics['raw_off_edp_per_drive'] - 
        raw_metrics['raw_def_edp_per_drive']
    ).round(3)
    
    # Calculate weighted metrics using the weighted weekly data
    # For each team, normalize weights to sum to number of non-zero weights
    weighted_metrics_grouped = weighted_metrics.groupby('team')
    games_played = weighted_metrics_grouped.apply(lambda x: (x['weight'] > 0).sum())
    weight_sums = weighted_metrics_grouped['weight'].sum()
    
    # Create normalization factors
    normalization = pd.DataFrame({
        'team': games_played.index,
        'games': games_played.values,
        'weight_sum': weight_sums.values
    })
    
    # Merge normalization factors back to weighted_metrics
    weighted_metrics = pd.merge(weighted_metrics, normalization, on='team')
    weighted_metrics['normalized_weight'] = weighted_metrics['weight'] * (
        weighted_metrics['games'] / weighted_metrics['weight_sum']
    )
    
    weighted_totals = weighted_metrics.groupby('team').agg({
        'earned_drive_points_off': lambda x: (x * weighted_metrics['normalized_weight']).sum(),
        'earned_drive_points_def': lambda x: (x * weighted_metrics['normalized_weight']).sum(),
        'drive_count_off': lambda x: (x * weighted_metrics['normalized_weight']).sum(),
        'drive_count_def': lambda x: (x * weighted_metrics['normalized_weight']).sum()
    }).reset_index()
    
    # Calculate weighted total metrics first
    weighted_totals['off_edp_weighted'] = weighted_totals['earned_drive_points_off'].apply(round_appropriately)
    weighted_totals['def_edp_weighted'] = weighted_totals['earned_drive_points_def'].apply(round_appropriately)
    weighted_totals['total_edp_weighted'] = (
        weighted_totals['off_edp_weighted'] - 
        weighted_totals['def_edp_weighted']
    ).apply(round_appropriately)
    
    # Then calculate weighted per-drive metrics
    weighted_totals['off_edp_weighted_per_drive'] = (
        weighted_totals['off_edp_weighted'] / 
        weighted_totals['drive_count_off']
    ).round(3)
    weighted_totals['def_edp_weighted_per_drive'] = (
        weighted_totals['def_edp_weighted'] / 
        weighted_totals['drive_count_def']
    ).round(3)
    weighted_totals['total_edp_weighted_per_drive'] = (
        weighted_totals['off_edp_weighted_per_drive'] - 
        weighted_totals['def_edp_weighted_per_drive']
    ).round(3)
    
    # Calculate SoS adjustments using the new iterative method
    # Separate offensive and defensive dataframes
    off = all_weekly_metrics.groupby(['team', 'game_id', 'week']).agg({
        'earned_drive_points_off': 'first',
        'drive_count_off': 'first'
    }).reset_index()
    
    def_ = all_weekly_metrics.groupby(['team', 'game_id', 'week']).agg({
        'earned_drive_points_def': 'first',
        'drive_count_def': 'first'
    }).reset_index()
    
    # Calculate iterative SoS adjustments on raw data
    sos_adjusted = calculator.calculate_strength_adjusted_edp(off, def_)
    
    # Merge in recency weights
    sos_adjusted = pd.merge(
        sos_adjusted,
        weighted_metrics[['team', 'game_id', 'week', 'normalized_weight']],
        on=['team', 'game_id', 'week'],
        how='left'
    )
    sos_adjusted['normalized_weight'] = sos_adjusted['normalized_weight'].fillna(0)
    
    # Calculate weighted SoS-adjusted metrics
    sos_totals = sos_adjusted.groupby('team').agg({
        'adjusted_edp_off': lambda x: (x * sos_adjusted['normalized_weight']).sum(),
        'adjusted_edp_def': lambda x: (x * sos_adjusted['normalized_weight']).sum(),
        'drive_count_off': lambda x: (x * sos_adjusted['normalized_weight']).sum(),
        'drive_count_def': lambda x: (x * sos_adjusted['normalized_weight']).sum()
    }).reset_index()
    
    # Calculate SoS adjusted total metrics first
    sos_totals['off_edp_SoS_adj'] = sos_totals['adjusted_edp_off'].apply(round_appropriately)
    sos_totals['def_edp_SoS_adj'] = sos_totals['adjusted_edp_def'].apply(round_appropriately)
    sos_totals['total_edp_SoS_adj'] = (
        sos_totals['off_edp_SoS_adj'] - 
        sos_totals['def_edp_SoS_adj']
    ).apply(round_appropriately)
    
    # Then calculate SoS adjusted per-drive metrics
    sos_totals['off_edp_SoS_adj_per_drive'] = (
        sos_totals['off_edp_SoS_adj'] / 
        sos_totals['drive_count_off']
    ).round(3)
    sos_totals['def_edp_SoS_adj_per_drive'] = (
        sos_totals['def_edp_SoS_adj'] / 
        sos_totals['drive_count_def']
    ).round(3)
    sos_totals['total_edp_SoS_adj_per_drive'] = (
        sos_totals['off_edp_SoS_adj_per_drive'] - 
        sos_totals['def_edp_SoS_adj_per_drive']
    ).round(3)
    
    # Merge all metrics
    season_metrics = pd.merge(raw_metrics, weighted_totals, on='team', how='outer', suffixes=('', '_y'))
    season_metrics = pd.merge(season_metrics, sos_totals, on='team', how='outer', suffixes=('', '_y'))
    
    # Drop duplicate columns
    cols_to_drop = [col for col in season_metrics.columns if col.endswith('_y')]
    season_metrics = season_metrics.drop(columns=cols_to_drop)
    
    # Reorder columns
    column_order = [
        'team',
        # Total EDP metrics
        'raw_total_edp',
        'total_edp_weighted',
        'total_edp_SoS_adj',
        'raw_edp_per_drive',
        'total_edp_weighted_per_drive',
        'total_edp_SoS_adj_per_drive',
        # Offensive EDP metrics
        'raw_off_edp',
        'off_edp_weighted',
        'off_edp_SoS_adj',
        'raw_off_edp_per_drive',
        'off_edp_weighted_per_drive',
        'off_edp_SoS_adj_per_drive',
        # Defensive EDP metrics
        'raw_def_edp',
        'def_edp_weighted',
        'def_edp_SoS_adj',
        'raw_def_edp_per_drive',
        'def_edp_weighted_per_drive',
        'def_edp_SoS_adj_per_drive',
        # Drive counts at the end
        'drive_count_off',
        'drive_count_def'
    ]
    
    season_metrics = season_metrics[column_order].round(3)
    
    return all_weekly_metrics, season_metrics

def save_results(weekly_metrics: pd.DataFrame, season_metrics: pd.DataFrame, week: int):
    """Save results to Excel with multiple sheets."""
    # Create output directory if it doesn't exist
    output_dir = Path('outputs')
    output_dir.mkdir(exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    filename = f"edp_rankings_week{week}_{timestamp}.xlsx"
    filepath = output_dir / filename
    
    # Save to Excel
    with pd.ExcelWriter(filepath) as writer:
        # Season totals (adjusted)
        season_metrics.sort_values(
            'total_edp_SoS_adj',
            ascending=False
        ).to_excel(writer, sheet_name='Season Rankings', index=False)
        
        # Latest week
        latest_week = weekly_metrics[
            weekly_metrics['week'] == week
        ].sort_values(
            'total_edp_per_drive',
            ascending=False
        ).to_excel(writer, sheet_name=f'Week {week}', index=False)
        
        # All weeks
        weekly_metrics.sort_values(
            ['week', 'total_edp_per_drive'],
            ascending=[False, False]
        ).to_excel(writer, sheet_name='All Weeks', index=False)
    
    print(f"\nResults saved to: {filepath}")

def main():
    """Main execution function."""
    # Load data
    print("\nLoading play-by-play data...")
    df = load_and_prepare_data()
    
    # Get latest week
    max_week = df['week'].max()
    print(f"\nAnalyzing through Week {max_week}...")
    
    # Calculate all metrics
    weekly_metrics, season_metrics = calculate_season_metrics(df, max_week)
    
    if not weekly_metrics.empty:
        save_results(weekly_metrics, season_metrics, max_week)
    else:
        print("No results to save.")

if __name__ == "__main__":
    main() 