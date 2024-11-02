# edp_analysis/ranking.py
import pandas as pd

def rank_teams_by_edp(total_edp, most_recent_week):
    # Filter for the most recent week of data
    game_edp_scores_recent = total_edp[total_edp['game_id'].str.contains(f'2024_0[1-{most_recent_week}]')]
    
    # Aggregate EDP data by team
    team_rankings = game_edp_scores_recent.groupby('team').agg(
        total_edp=('total_edp', 'sum'),
        offensive_edp=('offensive_edp', 'sum'),
        defensive_edp=('defensive_edp', 'sum'),
        offensive_edp_per_drive=('edp_per_drive', 'mean'),  # Offensive EDP per drive
        defensive_edp_per_drive=('defensive_edp_per_drive', 'mean'),  # Defensive EDP per drive
        total_edp_per_drive=('total_edp_per_drive', 'mean')  # Total EDP per drive
    ).reset_index()

    # Round EDP values to 1 decimal place
    team_rankings = team_rankings.round({
        'total_edp': 1,
        'offensive_edp': 1,
        'defensive_edp': 1,
        'offensive_edp_per_drive': 1,
        'defensive_edp_per_drive': 1,
        'total_edp_per_drive': 1
    })

    # Add rank columns and sort by total EDP
    team_rankings['total_rank'] = team_rankings['total_edp'].rank(ascending=False).round(1)
    team_rankings['offensive_rank'] = team_rankings['offensive_edp'].rank(ascending=False).round(1)
    team_rankings['defensive_rank'] = team_rankings['defensive_edp'].rank(ascending=True).round(1)  # lower defensive EDP is better

    # Sort the table by the total EDP rank
    team_rankings = team_rankings.sort_values(by='total_rank')

    # Return the final DataFrame
    return team_rankings

def save_rankings_image(team_rankings, file_name):
    # Create a styled table to be saved as an image
    styled_table = team_rankings.style.background_gradient(subset=['total_rank'], cmap='Blues_r') \
        .background_gradient(subset=['offensive_rank'], cmap='Greens_r') \
        .background_gradient(subset=['defensive_rank'], cmap='Purples_r')

    # Convert styled DataFrame to an image
    styled_table.to_excel(file_name.replace('.png', '.xlsx'), engine='openpyxl')
    print(f"Rankings saved as Excel file: {file_name.replace('.png', '.xlsx')}")
