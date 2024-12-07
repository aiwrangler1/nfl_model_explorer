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

from .edp_calculator import EDPCalculator
from .opponent_strength import OpponentStrengthAdjuster
from .game_prediction import EDPGamePredictor

class MatchupAnalyzer:
    def __init__(self, season: int = 2024):
        """
        Initialize the matchup analyzer.
        
        Args:
            season: NFL season to analyze (default: 2024)
        """
        self.season = season
        self.calculator = EDPCalculator()
        self.adjuster = OpponentStrengthAdjuster()
        self.predictor = EDPGamePredictor()
        
        # EDP weighting parameters
        self.recent_games_weight = 0.55  # Weight for most recent 3 games
        self.season_decay = 0.7  # Decay factor between seasons
        self.game_decay = 0.85  # Decay factor within season
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize model with historical data
        self._initialize_model()
    
    def calculate_weighted_edp(
        self,
        team_games: pd.DataFrame,
        current_season: int,
        metric_col: str
    ) -> float:
        """
        Calculate weighted EDP using exponential decay and season weights.
        
        Args:
            team_games: DataFrame with team's games sorted by date
            current_season: The season we're analyzing
            metric_col: Column name for the metric to weight
            
        Returns:
            Weighted EDP value
        """
        if len(team_games) == 0:
            return 0.0
            
        # Sort by season and week
        team_games = team_games.sort_values(['season', 'week'], ascending=True)
        
        # Debug print
        print(f"\nCalculating weighted EDP for metric: {metric_col}")
        print(f"Number of games: {len(team_games)}")
        print(f"Columns available: {team_games.columns.tolist()}")
        print(f"Sample of data:")
        print(team_games[[metric_col, 'season', 'week']].head())
        
        # Calculate weights
        weights = []
        total_weight = 0
        
        for idx, game in enumerate(team_games.itertuples()):
            # Season weight
            season_weight = 1.0 if game.season == current_season else self.season_decay ** (current_season - game.season)
            
            # Recent games weight (last 3 played games get higher weight)
            recency_weight = self.recent_games_weight if idx >= len(team_games) - 3 else 1.0
            
            # Game decay weight (exponential decay within season)
            game_weight = self.game_decay ** (len(team_games) - idx - 1)
            
            # Combine weights
            weight = season_weight * recency_weight * game_weight
            weights.append(weight)
            total_weight += weight
            
            # Debug print for first few games
            if idx < 3:
                print(f"\nGame {idx + 1}:")
                print(f"Season weight: {season_weight}")
                print(f"Recency weight: {recency_weight}")
                print(f"Game weight: {game_weight}")
                print(f"Combined weight: {weight}")
                print(f"Metric value: {getattr(game, metric_col)}")
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Calculate weighted average
        weighted_edp = sum(w * getattr(game, metric_col) 
                         for w, game in zip(weights, team_games.itertuples()))
        
        print(f"\nFinal weighted EDP: {weighted_edp}")
        
        return weighted_edp
    
    def compute_team_metrics(self) -> pd.DataFrame:
        """Compute EDP metrics for all teams."""
        try:
            # Load both 2023 and 2024 data
            df_2024 = self.calculator.load_and_process_data([2024])
            df_2023 = self.calculator.load_and_process_data([2023])
            
            # Print unique teams for debugging
            print("\nAvailable teams in 2024:")
            print(sorted(df_2024['posteam'].unique()))
            
            # Calculate metrics for both seasons
            offensive_edp_2024 = self.calculator.calculate_offensive_edp(df_2024)
            defensive_edp_2024 = self.calculator.calculate_defensive_edp(df_2024)
            offensive_edp_2023 = self.calculator.calculate_offensive_edp(df_2023)
            defensive_edp_2023 = self.calculator.calculate_defensive_edp(df_2023)
            
            # Combine seasons
            offensive_edp = pd.concat([offensive_edp_2023, offensive_edp_2024])
            defensive_edp = pd.concat([defensive_edp_2023, defensive_edp_2024])
            
            # Calculate weighted metrics first
            features = []
            for team in offensive_edp['team'].unique():
                # Get team's games
                team_off = offensive_edp[offensive_edp['team'] == team].copy()
                team_def = defensive_edp[defensive_edp['team'] == team].copy()
                
                # Calculate weighted metrics
                weighted_off_edp = self.calculate_weighted_edp(
                    team_off, self.season, 'offensive_edp'
                )
                weighted_def_edp = self.calculate_weighted_edp(
                    team_def, self.season, 'defensive_edp'
                )
                
                # Get games played this season
                games_2024 = len(team_off[team_off['season'] == 2024])
                
                # Get most recent game_id
                latest_game = team_off.sort_values(['season', 'week']).iloc[-1]
                
                features.append({
                    'team': team,
                    'game_id': latest_game['game_id'],
                    'season': latest_game['season'],
                    'week': latest_game['week'],
                    'offensive_edp': weighted_off_edp,
                    'defensive_edp': weighted_def_edp,
                    'games_played_2024': games_2024,
                    'data_source': '2024' if games_2024 > 0 else '2023'
                })
            
            features_df = pd.DataFrame(features)
            
            print("\nFeatures before opponent adjustments:")
            print(features_df[['team', 'offensive_edp', 'defensive_edp']].head())
            
            # Now apply opponent adjustments
            adjusted_metrics = self.adjuster.calculate_adjusted_metrics(
                offensive_edp=features_df[['team', 'game_id', 'season', 'week', 'offensive_edp']].rename(columns={'offensive_edp': 'offensive_edp'}),
                defensive_edp=features_df[['team', 'game_id', 'season', 'week', 'defensive_edp']].rename(columns={'defensive_edp': 'defensive_edp'})
            )
            
            print("\nAdjusted metrics:")
            print("Offensive:", adjusted_metrics['offensive'])
            print("Defensive:", adjusted_metrics['defensive'])
            
            # Add adjusted metrics back to features
            features_df['adjusted_offensive_edp'] = features_df['team'].map(adjusted_metrics['offensive'])
            features_df['adjusted_defensive_edp'] = features_df['team'].map(adjusted_metrics['defensive'])
            
            # Calculate total EDP
            features_df['total_edp'] = features_df['adjusted_offensive_edp'] - features_df['adjusted_defensive_edp']
            
            print("\nFeatures after opponent adjustments:")
            print(features_df[['team', 'adjusted_offensive_edp', 'adjusted_defensive_edp', 'total_edp']].head())
            
            logging.info(f"Using 2024 data for {sum(features_df['data_source'] == '2024')} teams, "
                        f"2023 data for {sum(features_df['data_source'] == '2023')} teams")
            
            return features_df
            
        except Exception as e:
            logging.error(f"Error computing metrics: {str(e)}")
            raise
    
    def analyze_matchup(
        self,
        home_team: str,
        away_team: str,
        neutral_site: bool = False
    ) -> Dict:
        """
        Analyze a specific matchup between two teams.
        
        Args:
            home_team: Home team abbreviation (e.g., 'BUF')
            away_team: Away team abbreviation (e.g., 'KC')
            neutral_site: Whether the game is at a neutral site
            
        Returns:
            Dictionary containing matchup analysis
        """
        try:
            # Get team metrics
            team_metrics = self.compute_team_metrics()
            
            # Verify teams exist
            if home_team not in team_metrics['team'].values:
                raise ValueError(f"Home team '{home_team}' not found in data")
            if away_team not in team_metrics['team'].values:
                raise ValueError(f"Away team '{away_team}' not found in data")
            
            # Get team-specific metrics
            home_metrics = team_metrics[team_metrics['team'] == home_team].iloc[0]
            away_metrics = team_metrics[team_metrics['team'] == away_team].iloc[0]
            
            # Create feature matrix for prediction
            X = pd.DataFrame({
                'game_id': ['matchup_1'],
                'home_team': [home_team],
                'away_team': [away_team],
                'edp_diff': home_metrics['adjusted_offensive_edp'] - away_metrics['adjusted_defensive_edp'],
                'def_edp_diff': home_metrics['adjusted_defensive_edp'] - away_metrics['adjusted_offensive_edp'],
                'home_edp_uncertainty': [0.1],  # Default uncertainty
                'away_edp_uncertainty': [0.1]   # Default uncertainty
            })
            
            # Generate predictions
            pred_mean, pred_lower, pred_upper = self.predictor.predict(
                X,
                return_intervals=True
            )
            
            # Calculate win probability
            win_prob = 1 / (1 + np.exp(-pred_mean[0] / 100))
            
            # Compile analysis
            analysis = {
                'home_team': home_team,
                'away_team': away_team,
                'predicted_spread': pred_mean[0],
                'confidence_interval': (pred_lower[0], pred_upper[0]),
                'home_win_prob': win_prob,
                'away_win_prob': 1 - win_prob,
                'team_metrics': {
                    home_team: {
                        'offensive_edp': home_metrics['adjusted_offensive_edp'],
                        'defensive_edp': home_metrics['adjusted_defensive_edp'],
                        'total_edp': home_metrics['total_edp'],
                        'games_played': home_metrics['games_played_2024']
                    },
                    away_team: {
                        'offensive_edp': away_metrics['adjusted_offensive_edp'],
                        'defensive_edp': away_metrics['adjusted_defensive_edp'],
                        'total_edp': away_metrics['total_edp'],
                        'games_played': away_metrics['games_played_2024']
                    }
                }
            }
            
            return analysis
            
        except Exception as e:
            logging.error(f"Error analyzing matchup: {str(e)}")
            raise
    
    def _initialize_model(self):
        """Initialize the prediction model with historical data."""
        try:
            # Load 2023 data for initialization
            historical_df = self.calculator.load_and_process_data([2023])
            
            # Get game-level points
            game_points = historical_df.groupby(['game_id', 'posteam'])['points'].sum().reset_index()
            
            # Calculate historical EDP metrics
            offensive_edp = self.calculator.calculate_offensive_edp(historical_df)
            defensive_edp = self.calculator.calculate_defensive_edp(historical_df)
            
            # Apply opponent adjustments
            adjusted_metrics = self.adjuster.calculate_adjusted_metrics(
                offensive_edp=offensive_edp,
                defensive_edp=defensive_edp
            )
            
            # Create training data
            train_data = []
            for game_id in offensive_edp['game_id'].unique():
                game_off = offensive_edp[offensive_edp['game_id'] == game_id]
                game_def = defensive_edp[defensive_edp['game_id'] == game_id]
                game_score = game_points[game_points['game_id'] == game_id]
                
                if len(game_off) == 2 and len(game_def) == 2 and len(game_score) == 2:
                    home_team = game_off.iloc[0]
                    away_team = game_off.iloc[1]
                    home_score = game_score[game_score['posteam'] == home_team['team']]['points'].iloc[0]
                    away_score = game_score[game_score['posteam'] == away_team['team']]['points'].iloc[0]
                    
                    train_data.append({
                        'game_id': game_id,
                        'home_team': home_team['team'],
                        'away_team': away_team['team'],
                        'edp_diff': adjusted_metrics['offensive'].get(home_team['team'], 0) - 
                                  adjusted_metrics['defensive'].get(away_team['team'], 0),
                        'def_edp_diff': adjusted_metrics['defensive'].get(home_team['team'], 0) - 
                                      adjusted_metrics['offensive'].get(away_team['team'], 0),
                        'home_edp_uncertainty': 0.1,
                        'away_edp_uncertainty': 0.1,
                        'point_differential': home_score - away_score
                    })
            
            # Convert to DataFrame
            train_df = pd.DataFrame(train_data)
            
            # Fit the model
            X_train = train_df.drop('point_differential', axis=1)
            y_train = train_df['point_differential'].values
            
            self.predictor.fit_hierarchical_model(X_train, y_train)
            
            logging.info("Model initialized with historical data")
            
        except Exception as e:
            logging.error(f"Error initializing model: {str(e)}")
            raise

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