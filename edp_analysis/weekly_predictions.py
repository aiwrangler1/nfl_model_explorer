"""
Direct NFL matchup predictions using EDP metrics.

This module enables head-to-head matchup analysis using:
1. Team-specific EDP metrics (with rolling windows and exponential decay)
2. Opponent adjustments
3. Bayesian prediction model
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

from .edp_calculator import EDPCalculator
from .opponent_strength import OpponentStrengthAdjuster
from .game_prediction import EDPGamePredictor

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Path('predictions/prediction_log.txt')),
        logging.StreamHandler()
    ]
)

VALID_TEAMS = {
    'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
    'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC',
    'LA', 'LAC', 'LV', 'MIA', 'MIN', 'NE', 'NO', 'NYG',
    'NYJ', 'PHI', 'PIT', 'SEA', 'SF', 'TB', 'TEN', 'WAS'
}

def validate_team(team: str) -> str:
    """Validate and normalize team abbreviation."""
    team = team.upper()
    if team not in VALID_TEAMS:
        raise ValueError(f"Invalid team abbreviation: {team}")
    return team

class MatchupAnalyzer:
    def __init__(self, season: int = 2024):
        """
        Initialize the matchup analyzer.
        
        Args:
            season: NFL season to analyze
        """
        self.season = season
        self.edp_calculator = EDPCalculator()
        self.strength_adjuster = OpponentStrengthAdjuster()
        self.game_predictor = EDPGamePredictor()
        
    def prepare_matchup_data(self, home_team: str, away_team: str) -> pd.DataFrame:
        """
        Prepare EDP metrics for a specific matchup.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            
        Returns:
            DataFrame with matchup metrics
        """
        # Validate teams
        home_team = validate_team(home_team)
        away_team = validate_team(away_team)
        
        # Calculate season-to-date metrics
        play_data = self.edp_calculator.load_and_process_data([self.season])
        total_edp = self.edp_calculator.calculate_total_edp(play_data)
        
        # Get latest metrics for both teams
        home_metrics = total_edp[total_edp['team'] == home_team].iloc[-1]
        away_metrics = total_edp[total_edp['team'] == away_team].iloc[-1]
        
        # Create matchup DataFrame
        matchup_data = pd.DataFrame({
            'home_team': [home_team],
            'away_team': [away_team],
            'home_offensive_edp': [home_metrics['offensive_edp']],
            'home_defensive_edp': [home_metrics['defensive_edp']],
            'home_st_edp': [home_metrics['st_edp']],
            'home_total_edp': [home_metrics['total_edp']],
            'away_offensive_edp': [away_metrics['offensive_edp']],
            'away_defensive_edp': [away_metrics['defensive_edp']],
            'away_st_edp': [away_metrics['st_edp']],
            'away_total_edp': [away_metrics['total_edp']]
        })
        
        return matchup_data
        
    def predict_matchup(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Predict the outcome of a matchup using EDP metrics.
        
        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            
        Returns:
            Dictionary with prediction probabilities and point spread
        """
        matchup_data = self.prepare_matchup_data(home_team, away_team)
        
        # Calculate matchup advantages
        matchup_data['offensive_advantage'] = (
            matchup_data['home_offensive_edp'] - matchup_data['away_defensive_edp']
        )
        matchup_data['defensive_advantage'] = (
            matchup_data['away_offensive_edp'] - matchup_data['home_defensive_edp']
        )
        matchup_data['st_advantage'] = (
            matchup_data['home_st_edp'] - matchup_data['away_st_edp']
        )
        
        # Get prediction from game predictor
        prediction = self.game_predictor.predict_game(
            matchup_data['offensive_advantage'].iloc[0],
            matchup_data['defensive_advantage'].iloc[0],
            matchup_data['st_advantage'].iloc[0]
        )
        
        return {
            'home_win_prob': prediction['home_win_prob'],
            'predicted_spread': prediction['spread'],
            'total_edp_diff': (
                matchup_data['home_total_edp'].iloc[0] - 
                matchup_data['away_total_edp'].iloc[0]
            )
        }

def print_matchup_analysis(analysis: Dict) -> None:
    """Print formatted matchup analysis."""
    print("\nMatchup Analysis")
    print("================")
    print(f"{analysis['home_team']} (Home) vs {analysis['away_team']} (Away)")
    
    print("\nSeason-to-Date Metrics:")
    print(f"{analysis['home_team']}:")
    print(f"  Games Played: {analysis['team_metrics'][analysis['home_team']]['games_played']}")
    print(f"  Offensive EDP: {analysis['team_metrics'][analysis['home_team']]['offensive_edp']:.2f}")
    print(f"  Defensive EDP: {analysis['team_metrics'][analysis['home_team']]['defensive_edp']:.2f}")
    print(f"  Total EDP: {analysis['team_metrics'][analysis['home_team']]['total_edp']:.2f}")
    
    print(f"\n{analysis['away_team']}:")
    print(f"  Games Played: {analysis['team_metrics'][analysis['away_team']]['games_played']}")
    print(f"  Offensive EDP: {analysis['team_metrics'][analysis['away_team']]['offensive_edp']:.2f}")
    print(f"  Defensive EDP: {analysis['team_metrics'][analysis['away_team']]['defensive_edp']:.2f}")
    print(f"  Total EDP: {analysis['team_metrics'][analysis['away_team']]['total_edp']:.2f}")
    
    print("\nGame Prediction:")
    print(f"Spread: {analysis['predicted_spread']:.1f} (negative means home team favored)")
    print(f"95% Confidence Interval: ({analysis['confidence_interval'][0]:.1f}, {analysis['confidence_interval'][1]:.1f})")
    print(f"Win Probabilities:")
    print(f"  {analysis['home_team']}: {analysis['home_win_prob']*100:.1f}%")
    print(f"  {analysis['away_team']}: {analysis['away_win_prob']*100:.1f}%")

def main():
    """Run matchup analysis for user-specified teams."""
    analyzer = MatchupAnalyzer(season=2024)
    
    # Get team inputs
    print("\nEnter team abbreviations (e.g., BUF, KC):")
    home_team = input("Home Team: ").upper()
    away_team = input("Away Team: ").upper()
    
    # Map team names
    team_map = {
        'LAR': 'LA',  # Los Angeles Rams
        'JAC': 'JAX',  # Jacksonville Jaguars
        'LAS': 'LV',  # Las Vegas Raiders
        'LAC': 'LAC'   # Los Angeles Chargers
    }
    
    home_team = team_map.get(home_team, home_team)
    away_team = team_map.get(away_team, away_team)
    
    # Run analysis
    try:
        analysis = analyzer.analyze_matchup(home_team, away_team)
        print_matchup_analysis(analysis)
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 