#!/usr/bin/env python3
"""
NFL Weekly EDP Analysis Script

This script performs weekly Expected Drive Points (EDP) analysis for NFL teams.
It loads play-by-play data, calculates EDP metrics, and generates team rankings.

Usage:
    python run_weekly_analysis.py [--week WEEK]
    
    --week: Optional. Specify which week to analyze through (1-18).
            If not provided, will use the latest week with available data.
"""

import sys
import os
import datetime
import argparse
from typing import List, Dict

import pandas as pd

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from edp_analysis.data_loader import load_and_prepare_data
from edp_analysis.edp_calculator import preprocess_data, calculate_team_edp, merge_edp
from edp_analysis.ranking import rank_teams_by_edp, save_rankings_excel


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Namespace containing the parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Run NFL EDP analysis for a specific week or latest available week."
    )
    parser.add_argument(
        "--week",
        type=int,
        help="Week to analyze through (1-18). If not provided, uses latest available week.",
        choices=range(1, 19),
        metavar="WEEK"
    )
    return parser.parse_args()


def calculate_team_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate additional per-team metrics including EP-based metrics."""
    team_metrics: Dict = {}
    
    for team in df['posteam'].unique():
        team_data = df[df['posteam'] == team]
        
        # Calculate drive metrics
        total_drives = team_data['drive'].nunique()
        
        # Calculate EP-based metrics
        drive_ep_data = team_data.groupby('drive').agg({
            'ep': lambda x: x.iloc[-1] - x.iloc[0],  # EP gained in drive
            'earned_drive_points': 'sum'  # Points scored in drive
        }).fillna(0)
        
        ep_gained = drive_ep_data['ep'].mean()
        drive_ep_efficiency = (drive_ep_data['ep'] / 
                             team_data.groupby('drive').size()).mean()
        
        # Calculate success metrics
        drives_with_points = (drive_ep_data['earned_drive_points'] > 0).sum()
        success_rate = drives_with_points / total_drives if total_drives > 0 else 0
        
        # Calculate points per drive
        total_points = calculate_total_points(df, team)
        points_per_drive = total_points / total_drives if total_drives > 0 else 0
        
        team_metrics[team] = {
            'ep_gained_per_drive': round(ep_gained, 3),
            'drive_ep_efficiency': round(drive_ep_efficiency, 3),
            'drive_success_rate': round(success_rate, 3),
            'points_per_drive': round(points_per_drive, 3)
        }
    
    return pd.DataFrame.from_dict(team_metrics, orient='index')


def calculate_total_points(df: pd.DataFrame, team: str) -> int:
    """Calculate total points scored by a team across all games."""
    total_points = 0
    for game_id in df[df['posteam'] == team]['game_id'].unique():
        game_df = df[df['game_id'] == game_id].iloc[0]
        total_points += (game_df['total_home_score'] if team == game_df['home_team'] 
                        else game_df['total_away_score'])
    return total_points


def create_team_rankings(df: pd.DataFrame, analysis_week: int) -> pd.DataFrame:
    """Create team rankings with EP metrics."""
    # Calculate EDP metrics
    offensive_edp, defensive_edp = calculate_team_edp(df)
    total_edp = merge_edp(offensive_edp, defensive_edp)
    
    # Calculate additional team metrics
    team_metrics = calculate_team_metrics(df)
    
    # Create and format rankings
    team_rankings = rank_teams_by_edp(total_edp, analysis_week)
    team_rankings = team_rankings.join(team_metrics)
    
    # Select and order columns
    columns_to_keep = [
        'team',
        'total_rank', 'offense_rank', 'defense_rank', 'ep_efficiency_rank',
        'off_edp_per_drive', 'weighted_off_edp_per_drive',
        'def_edp_per_drive', 'weighted_def_edp_per_drive',
        'total_edp_per_drive', 'weighted_total_edp_per_drive',
        'ep_gained_per_drive', 'drive_ep_efficiency',
        'drive_success_rate', 'points_per_drive'
    ]
    
    return team_rankings[columns_to_keep]


def save_rankings(rankings: pd.DataFrame, analysis_week: int) -> str:
    """
    Save team rankings to an Excel file.
    
    Args:
        rankings: DataFrame containing team rankings
        analysis_week: Current week of analysis
        
    Returns:
        Path to the saved Excel file
    """
    # Create output folder
    output_folder = os.path.join(project_root, "Weekly Ranking Outputs")
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%b %d-%I:%M%p").lower()
    excel_filename = os.path.join(
        output_folder,
        f"nfl_edp_rankings_2024_week_{analysis_week}_{timestamp}.xlsx"
    )
    
    # Save rankings
    save_rankings_excel(rankings, excel_filename)
    return excel_filename


def main() -> None:
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Define analysis parameters
    years = [2024]
    
    try:
        # Load and preprocess data
        df_processed, analysis_week = load_and_prepare_data(years, args.week)
        df_processed = preprocess_data(df_processed)
        
        # Create team rankings
        team_rankings = create_team_rankings(df_processed, analysis_week)
        
        # Save results
        output_file = save_rankings(team_rankings, analysis_week)
        
        print(f"\nTeam Rankings through Week {analysis_week} of {years[-1]}:")
        print(f"Rankings saved to Excel: {output_file}")
        
    except Exception as e:
        print(f"\nError running weekly analysis: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

    #!/usr/bin/env python3
"""
NFL Season-by-Week EDP Analysis Script

This script analyzes Expected Drive Points (EDP) metrics for each week individually,
plus a cumulative summary for the entire season.
"""

import sys
import os
import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from edp_analysis.data_loader import load_and_prepare_data
from edp_analysis.edp_calculator import preprocess_data, calculate_team_edp, merge_edp


def calculate_single_week_edp(df: pd.DataFrame, week: int) -> pd.DataFrame:
    """Calculate EDP metrics for a single week with EP consideration."""
    # Filter data for just this week
    week_mask = (df['game_id']
                .str.extract(r'2024_(\d+)')[0]
                .fillna(0)
                .astype(int) == week)
    df_week = df[week_mask].copy()
    
    # Ensure EP columns are present
    if 'ep' not in df_week.columns:
        print(f"Warning: EP data missing for week {week}")
        return None
    
    # Calculate EDP metrics
    offensive_edp, defensive_edp = calculate_team_edp(df_week)
    total_edp = merge_edp(offensive_edp, defensive_edp)
    
    # Add EP efficiency metrics
    total_edp['ep_efficiency'] = (
        total_edp['weighted_off_edp_per_drive'] / 
        total_edp['weighted_def_edp_per_drive']
    ).fillna(0)
    
    # Add week number
    total_edp['week'] = week
    
    return total_edp


def calculate_cumulative_edp(df: pd.DataFrame, max_week: int) -> pd.DataFrame:
    """
    Calculate cumulative EDP metrics through the latest week.
    
    Args:
        df: Preprocessed play-by-play data
        max_week: Latest week to include
        
    Returns:
        DataFrame with cumulative team metrics
    """
    # Filter data through max week
    week_mask = (df['game_id']
                .str.extract(r'2024_(\d+)')[0]
                .fillna(0)
                .astype(int) <= max_week)
    print(f"Week Mask Type: {type(week_mask)}")  # Debugging output
    df_through_week = df[week_mask]
    print(f"df_through_week Type: {type(df_through_week)}")  # Debugging output
    
    # Calculate cumulative EDP metrics
    edp_results = calculate_team_edp(df_through_week)
    print(f"EDP Results Type: {type(edp_results)}")  # Debugging output
    if isinstance(edp_results, tuple):
        offensive_edp, defensive_edp = edp_results
    else:
        offensive_edp, defensive_edp = edp_results, None
    
    total_edp = merge_edp(offensive_edp, defensive_edp)
    
    return total_edp


def process_season_data(df: pd.DataFrame, max_week: int) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
    """Process EDP data with EP metrics for all weeks."""
    weekly_rankings = []
    
    # Calculate individual week stats
    for week in range(1, max_week + 1):
        print(f"\nProcessing Week {week}...")
        week_rankings = calculate_single_week_edp(df, week)
        if week_rankings is not None:
            weekly_rankings.append(week_rankings)
    
    # Calculate cumulative stats
    print("\nCalculating season totals...")
    cumulative_rankings = calculate_cumulative_edp(df, max_week)
    
    # Add EP efficiency to cumulative rankings
    cumulative_rankings['ep_efficiency'] = (
        cumulative_rankings['weighted_off_edp_per_drive'] / 
        cumulative_rankings['weighted_def_edp_per_drive']
    ).fillna(0)
    
    return weekly_rankings, cumulative_rankings


def save_season_rankings(weekly_rankings: List[pd.DataFrame],
                        cumulative_rankings: pd.DataFrame,
                        output_folder: str) -> str:
    """
    Save weekly and cumulative rankings to Excel.
    
    Args:
        weekly_rankings: List of DataFrames with weekly stats
        cumulative_rankings: DataFrame with cumulative stats
        output_folder: Directory to save output
        
    Returns:
        Path to saved Excel file
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.datetime.now().strftime("%b %d-%I:%M%p").lower()
    max_week = len(weekly_rankings)
    excel_filename = os.path.join(
        output_folder,
        f"nfl_edp_rankings_2024_weeks_1-{max_week}_{timestamp}.xlsx"
    )
    
    # Save to Excel with a sheet for each week plus cumulative
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Save each individual week
        for week, week_data in enumerate(weekly_rankings, 1):
            # Sort by weighted total EDP
            week_data = week_data.sort_values('weighted_total_edp_per_drive', ascending=False)
            
            # Round numeric columns
            numeric_cols = week_data.select_dtypes(include=['float64']).columns
            week_data[numeric_cols] = week_data[numeric_cols].round(3)
            
            # Save to sheet
            week_data.to_excel(
                writer,
                sheet_name=f'Week {week}',
                index=False
            )
            
            # Format columns
            worksheet = writer.sheets[f'Week {week}']
            for column in worksheet.columns:
                max_length = 0
                column = [cell for cell in column]
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                    adjusted_width = (max_length + 2)
                    worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
        
        # Save cumulative stats
        cumulative_rankings = cumulative_rankings.sort_values(
            'weighted_total_edp_per_drive',
            ascending=False
        )
        
        # Round numeric columns
        numeric_cols = cumulative_rankings.select_dtypes(include=['float64']).columns
        cumulative_rankings[numeric_cols] = cumulative_rankings[numeric_cols].round(3)
        
        # Save to sheet
        cumulative_rankings.to_excel(
            writer,
            sheet_name='Season Totals',
            index=False
        )
        
        # Format cumulative sheet
        worksheet = writer.sheets['Season Totals']
        for column in worksheet.columns:
            max_length = 0
            column = [cell for cell in column]
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column[0].column_letter].width = adjusted_width
    
    return excel_filename


def main() -> None:
    """Main execution function."""
    try:
        # Load and preprocess data
        print("\nLoading and preprocessing data...")
        years = [2024]
        df_all_years = load_and_prepare_data(years)
        df_prepared = preprocess_data(df_all_years)
        
        # Determine latest week
        max_week = (df_prepared['game_id']
                   .str.extract(r'2024_(\d+)')[0]
                   .fillna(0)
                   .astype(int)
                   .max())
        
        print(f"\nAnalyzing data through Week {max_week}")
        
        # Process all weeks and calculate cumulative stats
        weekly_rankings, cumulative_rankings = process_season_data(df_prepared, max_week)
        
        # Save results
        output_folder = os.path.join(project_root, "Weekly Ranking Outputs")
        output_file = save_season_rankings(
            weekly_rankings,
            cumulative_rankings,
            output_folder
        )
        
        print(f"\nSeason rankings saved to: {output_file}")
        
    except Exception as e:
        print(f"\nError processing season data: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()