"""Visualization functions for NFL EDP analysis."""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def create_team_performance_chart(season_metrics: pd.DataFrame, week: int, output_dir: str = 'outputs/visuals'):
    """
    Create scatter plots of team offensive vs defensive EDP (SoS adjusted),
    both total and per-drive versions.
    
    Args:
        season_metrics: DataFrame containing team metrics
        week: Current week number
        output_dir: Directory to save the plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    print(f"\nGenerating visualization charts in {output_dir}/...")
    
    # Create both total and per-drive charts
    metrics = [
        {
            'x': 'off_edp_SoS_adj',
            'y': 'def_edp_SoS_adj',
            'title': f'Team EDP, Strength Adjusted (Through Week {week})',
            'filename': f'team_performance_total_week{week}.png',
            'is_per_drive': False,
            'tick_spacing': 50,  # Tick every 50 points for raw EDP
            'label_offset': 20   # Larger offset for raw EDP
        },
        {
            'x': 'off_edp_SoS_adj_per_drive',
            'y': 'def_edp_SoS_adj_per_drive',
            'title': f'Team EDP per Drive, Strength Adjusted (Through Week {week})',
            'filename': f'team_performance_per_drive_week{week}.png',
            'is_per_drive': True,
            'tick_spacing': 0.2,  # Tick every 0.2 points for per-drive
            'label_offset': 0.1   # Smaller offset for per-drive
        }
    ]
    
    for metric in metrics:
        plt.figure(figsize=(12, 8))
        
        # Create scatter plot
        plt.scatter(
            season_metrics[metric['x']],
            -season_metrics[metric['y']],  # Invert defensive EDP (negative = good)
            alpha=0.6
        )
        
        # Add team labels
        for _, row in season_metrics.iterrows():
            plt.annotate(
                row['team'],
                (row[metric['x']], -row[metric['y']]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        # Calculate league averages for this metric
        off_avg = season_metrics[metric['x']].mean()
        def_avg = -season_metrics[metric['y']].mean()  # Note the negative sign
        
        # Add quadrant lines at league average with labels
        plt.axhline(y=def_avg, color='#1f77b4', linestyle='--', alpha=0.6, linewidth=1.5)
        plt.axvline(x=off_avg, color='#2ca02c', linestyle='--', alpha=0.6, linewidth=1.5)
        
        # Get axis limits for label positioning
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        # Add league average labels with background
        # Defensive (horizontal) line label - positioned inside chart on the left
        def_label = f'League Avg Defense: {def_avg:.2f}'
        plt.annotate(
            def_label,
            xy=(x_min + (x_max - x_min) * 0.02, def_avg),  # Position 2% from left edge
            xytext=(0, 0),
            textcoords='offset points',
            ha='left',
            va='center',
            color='#1f77b4',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),
            weight='bold',
            fontsize=10
        )
        
        # Offensive (vertical) line label - positioned inside chart near top
        off_label = f'League Avg Offense: {off_avg:.2f}'
        plt.annotate(
            off_label,
            xy=(off_avg, y_max - metric['label_offset']),  # Use metric-specific offset
            xytext=(0, 0),
            textcoords='offset points',
            ha='center',
            va='bottom',
            color='#2ca02c',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8, pad=2),
            weight='bold',
            fontsize=10
        )
        
        # Labels and title
        plt.title(metric['title'], pad=20)
        plt.xlabel('Off. EDP')
        plt.ylabel('Def. EDP')
        
        # Add quadrant labels with background
        plt.text(x_min, y_min, 'Bad O, Bad D', 
                ha='left', va='bottom', alpha=0.7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        plt.text(x_max, y_min, 'Good O, Bad D',
                ha='right', va='bottom', alpha=0.7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        plt.text(x_min, y_max, 'Bad O, Good D',
                ha='left', va='top', alpha=0.7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        plt.text(x_max, y_max, 'Good O, Good D',
                ha='right', va='top', alpha=0.7,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Equal aspect ratio to make quadrants square
        plt.axis('equal')
        
        # Set tick marks based on the metric type
        x_min, x_max = plt.xlim()
        y_min, y_max = plt.ylim()
        
        # Calculate tick positions
        spacing = metric['tick_spacing']
        x_ticks = [i*spacing for i in range(int(x_min/spacing)-1, int(x_max/spacing)+2)]
        y_ticks = [i*spacing for i in range(int(y_min/spacing)-1, int(y_max/spacing)+2)]
        
        plt.xticks(x_ticks, rotation=45)
        plt.yticks(y_ticks)
        
        # Add grid
        plt.grid(True, alpha=0.2)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save plot
        output_path = output_dir / metric['filename']
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Created {output_path}")
        plt.close()
    
    print("Visualization complete!\n")

if __name__ == "__main__":
    # Example usage
    from edp_analysis import calculate_season_metrics
    from data_loader import load_and_prepare_data
    
    # Load data and calculate metrics
    df = load_and_prepare_data()
    max_week = df['week'].max()
    weekly_metrics, season_metrics = calculate_season_metrics(df, max_week)
    
    # Create charts
    create_team_performance_chart(season_metrics, max_week) 