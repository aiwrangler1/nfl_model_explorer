#!/usr/bin/env python3
"""
NFL EDP Visualization Script

This script creates visualizations for Expected Drive Points (EDP) metrics,
including a scatter plot of offensive vs defensive EDP per drive.

Note: Defensive EDP is interpreted as points prevented, so negative values are better
(e.g., -2.0 means the defense prevents 2 points per drive on average).
"""

import os
import glob
from typing import Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def get_latest_rankings_file(output_dir: str = 'model_outputs') -> Optional[str]:
    """Get the path to the most recent rankings Excel file."""
    pattern = os.path.join(output_dir, 'edp_rankings_week*.xlsx')
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by modification time, newest first
    return max(files, key=os.path.getmtime)

def create_edp_scatter_plot(rankings_file: str) -> None:
    """Create a scatter plot of offensive vs defensive EDP per drive.
    
    Note: Defensive EDP is plotted as is, where negative values indicate better defense
    (more points prevented per drive). Y-axis is inverted so better defenses appear at top.
    """
    # Read the season rankings
    df = pd.read_excel(rankings_file, sheet_name='Season Rankings')
    
    # Set up the plot style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot using seaborn
    ax = sns.scatterplot(
        data=df,
        x='offensive_edp_per_drive',
        y='defensive_edp_per_drive',
        alpha=0.6
    )
    
    # Invert y-axis so better defenses (negative values) are at the top
    ax.invert_yaxis()
    
    # Add team labels
    for idx, row in df.iterrows():
        ax.annotate(
            row['team'],
            (row['offensive_edp_per_drive'], row['defensive_edp_per_drive']),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=8,
            alpha=0.8
        )
    
    # Add quadrant lines at league averages
    off_avg = df['offensive_edp_per_drive'].mean()
    def_avg = df['defensive_edp_per_drive'].mean()
    
    plt.axvline(x=off_avg, color='gray', linestyle='--', alpha=0.3)
    plt.axhline(y=def_avg, color='gray', linestyle='--', alpha=0.3)
    
    # Customize plot
    plt.title('NFL Teams: Offensive vs Defensive EDP per Drive', pad=20)
    plt.xlabel('Offensive EDP per Drive')
    plt.ylabel('Defensive EDP per Drive')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    output_dir = os.path.dirname(rankings_file)
    output_path = os.path.join(output_dir, 'edp_scatter_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nScatter plot saved to: {output_path}")

def main():
    """Main execution function."""
    # Get latest rankings file
    rankings_file = get_latest_rankings_file()
    
    if not rankings_file:
        print("Error: No rankings files found in model_outputs directory")
        return
    
    print(f"\nUsing rankings file: {rankings_file}")
    
    # Create visualization
    create_edp_scatter_plot(rankings_file)

if __name__ == "__main__":
    main() 