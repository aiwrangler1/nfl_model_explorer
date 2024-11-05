import pandas as pd
import numpy as np
from scipy.stats import poisson
from edp_analysis.edp_calculator import (
    preprocess_data, 
    calculate_rolling_team_edp, 
    calculate_weighted_rolling_edp,
    calculate_strength_adjusted_edp,
    apply_home_away_adjustment
)

def filter_2024_data(df):
    """Filter data to only include 2024 season games"""
    return df[df['game_id'].str.startswith('2024')]

def analyze_matchup(home_team, away_team, team_stats):
    """
    Analyze a specific matchup between two teams (e.g., KC, MIA)
    
    Args:
        home_team: BUF
        away_team: MIA
        team_stats: DataFrame containing team EDP statistics
    """
    # Get latest team stats
    latest_stats = team_stats.groupby('team').last().reset_index()
    
    # Calculate expected points with home field advantage
    HOME_ADVANTAGE = 1.5  # NFL average home field advantage
    
    # Home team expected points
    home_off = latest_stats.loc[latest_stats['team'] == home_team, 'adjusted_offensive_edp'].values[0]
    away_def = latest_stats.loc[latest_stats['team'] == away_team, 'adjusted_defensive_edp'].values[0]
    home_exp_points = (home_off + away_def) / 2 + (HOME_ADVANTAGE / 2)
    
    # Away team expected points
    away_off = latest_stats.loc[latest_stats['team'] == away_team, 'adjusted_offensive_edp'].values[0]
    home_def = latest_stats.loc[latest_stats['team'] == home_team, 'adjusted_defensive_edp'].values[0]
    away_exp_points = (away_off + home_def) / 2 - (HOME_ADVANTAGE / 2)
    
    # Calculate score distributions
    home_dist = poisson_score_distribution(home_exp_points)
    away_dist = poisson_score_distribution(away_exp_points)
    
    # Calculate win probabilities
    win_prob, tie_prob = calculate_win_probability(home_dist, away_dist)
    
    return {
        'home_exp_points': home_exp_points,
        'away_exp_points': away_exp_points,
        'home_win_prob': win_prob,
        'tie_prob': tie_prob,
        'away_win_prob': 1 - win_prob - tie_prob
    }

def poisson_score_distribution(expected_points, max_points=8):
    """Calculate probability distribution of possible scores"""
    return [poisson.pmf(score, expected_points) for score in range(max_points)]

def calculate_win_probability(team1_dist, team2_dist):
    """Calculate win probability based on score distributions"""
    win_prob = 0
    tie_prob = 0
    
    for score1, prob1 in enumerate(team1_dist):
        for score2, prob2 in enumerate(team2_dist):
            joint_prob = prob1 * prob2
            if score1 > score2:
                win_prob += joint_prob
            elif score1 == score2:
                tie_prob += joint_prob
    
    return win_prob, tie_prob

def main():
    # Load preprocessed EDP data from edp_calculator output
    df = pd.read_csv('../data/processed/team_edp_metrics.csv')
    
    # Filter for 2024 data only
    df_2024 = filter_2024_data(df)
    
    # Input matchup teams
    home_team = input("Enter home team abbreviation (e.g., KC): ").upper()
    away_team = input("Enter away team abbreviation (e.g., SF): ").upper()
    
    # Analyze matchup
    results = analyze_matchup(home_team, away_team, df_2024)
    
    # Print results
    print(f"\nMatchup Analysis: {home_team} (Home) vs {away_team}")
    print(f"{home_team} Expected Points: {results['home_exp_points']:.2f}")
    print(f"{away_team} Expected Points: {results['away_exp_points']:.2f}")
    print(f"{home_team} Win Probability: {results['home_win_prob']:.2%}")
    print(f"Tie Probability: {results['tie_prob']:.2%}")
    print(f"{away_team} Win Probability: {results['away_win_prob']:.2%}")
    
    # Print additional team statistics
    print("\nTeam Statistics (2024):")
    team_stats = df_2024.groupby('team').last()
    for team in [home_team, away_team]:
        print(f"\n{team}:")
        print(f"Offensive EDP: {team_stats.loc[team, 'adjusted_offensive_edp']:.2f}")
        print(f"Defensive EDP: {team_stats.loc[team, 'adjusted_defensive_edp']:.2f}")

if __name__ == "__main__":
    main()
