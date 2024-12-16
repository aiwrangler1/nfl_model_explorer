"""
Advanced game prediction modeling using EDP metrics.

This module implements sophisticated statistical approaches for predicting NFL game outcomes
using EDP (Earned Drive Points) metrics. It employs Bayesian hierarchical models and
advanced regression techniques to generate robust predictions with uncertainty quantification.
"""

import numpy as np
import pandas as pd
from scipy import stats
import aesara
import pymc as pm
from typing import Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler
import warnings

class EDPGamePredictor:
    def __init__(
        self,
        model_type: str = "hierarchical_bayes",
        mcmc_samples: int = 2000,
        random_seed: int = 42
    ):
        """
        Initialize the EDP-based game prediction model.
        
        Args:
            model_type: Type of model to use ('hierarchical_bayes', 'robust_regression')
            mcmc_samples: Number of MCMC samples for Bayesian inference
            random_seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.mcmc_samples = mcmc_samples
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.model = None
        self.trace = None
        
    def prepare_matchup_features(
        self,
        df: pd.DataFrame,
        lookback_weeks: int = 8
    ) -> pd.DataFrame:
        """
        Prepare features for game prediction from raw EDP data.
        
        Args:
            df: DataFrame with team EDP metrics
            lookback_weeks: Number of weeks to look back for rolling metrics
            
        Returns:
            DataFrame with prepared features for each matchup
        """
        # Calculate rolling EDP metrics
        team_metrics = []
        for team in df['team'].unique():
            team_df = df[df['team'] == team].sort_values(['season', 'week'])
            
            # Calculate rolling averages and standard deviations
            rolling_metrics = pd.DataFrame({
                'team': team,
                'season': team_df['season'],
                'week': team_df['week'],
                'rolling_edp_mean': team_df['adjusted_offensive_edp'].rolling(lookback_weeks, min_periods=3).mean(),
                'rolling_edp_std': team_df['adjusted_offensive_edp'].rolling(lookback_weeks, min_periods=3).std(),
                'rolling_def_edp_mean': team_df['adjusted_defensive_edp'].rolling(lookback_weeks, min_periods=3).mean(),
                'rolling_def_edp_std': team_df['adjusted_defensive_edp'].rolling(lookback_weeks, min_periods=3).std()
            })
            
            # Fill missing values with team averages
            rolling_metrics['rolling_edp_mean'] = rolling_metrics['rolling_edp_mean'].fillna(team_df['adjusted_offensive_edp'].mean())
            rolling_metrics['rolling_def_edp_mean'] = rolling_metrics['rolling_def_edp_mean'].fillna(team_df['adjusted_defensive_edp'].mean())
            
            # Fill missing uncertainties with small default values
            rolling_metrics['rolling_edp_std'] = rolling_metrics['rolling_edp_std'].fillna(0.1)
            rolling_metrics['rolling_def_edp_std'] = rolling_metrics['rolling_def_edp_std'].fillna(0.1)
            
            team_metrics.append(rolling_metrics)
            
        team_metrics = pd.concat(team_metrics)
        
        # Create matchup features
        matchups = []
        for game_id in df['game_id'].unique():
            game_df = df[df['game_id'] == game_id]
            if len(game_df) != 2:
                continue
                
            home_team = game_df[game_df['home_away'] == 'home'].iloc[0]
            away_team = game_df[game_df['home_away'] == 'away'].iloc[0]
            
            home_metrics = team_metrics[
                (team_metrics['team'] == home_team['team']) &
                (team_metrics['season'] == home_team['season']) &
                (team_metrics['week'] == home_team['week'])
            ].iloc[0]
            
            away_metrics = team_metrics[
                (team_metrics['team'] == away_team['team']) &
                (team_metrics['season'] == away_team['season']) &
                (team_metrics['week'] == away_team['week'])
            ].iloc[0]
            
            matchup = {
                'game_id': game_id,
                'season': home_team['season'],
                'week': home_team['week'],
                'home_team': home_team['team'],
                'away_team': away_team['team'],
                'edp_diff': home_metrics['rolling_edp_mean'] - away_metrics['rolling_edp_mean'],
                'def_edp_diff': home_metrics['rolling_def_edp_mean'] - away_metrics['rolling_def_edp_mean'],
                'home_edp_uncertainty': home_metrics['rolling_edp_std'],
                'away_edp_uncertainty': away_metrics['rolling_edp_std'],
                'point_differential': home_team['points'] - home_team['opponent_points']
            }
            matchups.append(matchup)
            
        return pd.DataFrame(matchups)
    
    def predict_game(
        self,
        offensive_advantage: float,
        defensive_advantage: float,
        st_advantage: float,
        home_field: bool = True
    ) -> Dict[str, float]:
        """
        Predict game outcome using offensive, defensive, and special teams advantages.
        
        Args:
            offensive_advantage: Home offensive EDP - Away defensive EDP
            defensive_advantage: Away offensive EDP - Home defensive EDP
            st_advantage: Home ST_EDP - Away ST_EDP
            home_field: Whether to include home field advantage
            
        Returns:
            Dictionary with prediction probabilities and spread
        """
        # Base weights for each component
        OFF_WEIGHT = 0.45
        DEF_WEIGHT = 0.45
        ST_WEIGHT = 0.10
        HOME_ADVANTAGE = 2.5 if home_field else 0
        
        # Calculate weighted total advantage
        total_advantage = (
            OFF_WEIGHT * offensive_advantage +
            DEF_WEIGHT * defensive_advantage +
            ST_WEIGHT * st_advantage +
            HOME_ADVANTAGE
        )
        
        # Convert to spread (negative means home team favored)
        predicted_spread = -total_advantage * 7.0  # Convert EDP to points
        
        # Calculate win probability using logistic function
        home_win_prob = 1 / (1 + np.exp(-total_advantage))
        
        return {
            'home_win_prob': home_win_prob,
            'spread': predicted_spread,
            'total_advantage': total_advantage
        }
        
    def fit_hierarchical_model(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        include_st: bool = True
    ) -> None:
        """
        Fit hierarchical Bayesian model including special teams.
        
        Args:
            X: Feature matrix with EDP differences
            y: Target point differentials
            include_st: Whether to include special teams in model
        """
        with pm.Model() as model:
            # Priors for coefficients
            β_off = pm.Normal('β_off', mu=0.45, sigma=0.1)
            β_def = pm.Normal('β_def', mu=0.45, sigma=0.1)
            β_st = pm.Normal('β_st', mu=0.10, sigma=0.05)
            
            # Home field advantage
            hfa = pm.Normal('hfa', mu=2.5, sigma=1.0)
            
            # Model error
            σ = pm.HalfNormal('σ', sigma=10)
            
            # Expected point differential
            μ = (
                β_off * X['offensive_advantage'].values +
                β_def * X['defensive_advantage'].values +
                (β_st * X['st_advantage'].values if include_st else 0) +
                hfa
            )
            
            # Likelihood
            y_obs = pm.Normal('y_obs', mu=μ, sigma=σ, observed=y)
            
            # Fit model
            self.trace = pm.sample(
                self.mcmc_samples,
                tune=1000,
                return_inferencedata=True,
                random_seed=self.random_seed
            )
            
        self.model = model
    
    def predict(
        self,
        X_new: pd.DataFrame,
        return_intervals: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate predictions for new matchups.
        
        Args:
            X_new: New matchup features
            return_intervals: Whether to return prediction intervals
            
        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        # Scale features
        X_scaled = X_new.copy()
        feature_cols = ['edp_diff', 'def_edp_diff', 'home_edp_uncertainty', 'away_edp_uncertainty']
        X_scaled[feature_cols] = self.scaler.transform(X_new[feature_cols])
        
        if self.model_type == "hierarchical_bayes":
            with self.model:
                # Get posterior samples
                posterior = self.trace.posterior
                
                # Calculate predictions for each posterior sample
                team_idx = pd.Categorical(X_new['home_team']).codes
                
                # Extract parameters
                alpha = posterior['alpha'].values  # shape: (chains, draws, teams)
                beta_edp = posterior['beta_edp'].values  # shape: (chains, draws)
                beta_def = posterior['beta_def'].values  # shape: (chains, draws)
                sigma = posterior['sigma'].values  # shape: (chains, draws)
                
                # Reshape for broadcasting
                alpha = alpha.reshape(-1, alpha.shape[-1])  # shape: (chains*draws, teams)
                beta_edp = beta_edp.reshape(-1)  # shape: (chains*draws,)
                beta_def = beta_def.reshape(-1)  # shape: (chains*draws,)
                sigma = sigma.reshape(-1)  # shape: (chains*draws,)
                
                # Calculate predictions
                mu = (alpha[:, team_idx] + 
                      beta_edp[:, np.newaxis] * X_scaled['edp_diff'].values +
                      beta_def[:, np.newaxis] * X_scaled['def_edp_diff'].values)
                
                # Add noise for prediction intervals
                if return_intervals:
                    predictions = np.random.normal(
                        loc=mu,
                        scale=sigma[:, np.newaxis],
                        size=mu.shape
                    )
                else:
                    predictions = mu
                
                # Calculate summary statistics
                mean_pred = predictions.mean(axis=0) * self.y_scale
                
                if return_intervals:
                    lower = np.percentile(predictions, 2.5, axis=0) * self.y_scale
                    upper = np.percentile(predictions, 97.5, axis=0) * self.y_scale
                    return mean_pred, lower, upper
                
                return mean_pred, None, None
        
        else:
            raise NotImplementedError(f"Prediction not implemented for model type: {self.model_type}")
    
    def evaluate_calibration(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluate prediction calibration and accuracy.
        
        Args:
            X_test: Test set features
            y_test: True point differentials
            
        Returns:
            Dictionary of evaluation metrics
        """
        predictions, lower, upper = self.predict(X_test, return_intervals=True)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(np.mean((predictions - y_test) ** 2)),
            'mae': np.mean(np.abs(predictions - y_test)),
            'coverage_95': np.mean((y_test >= lower) & (y_test <= upper)),
            'interval_width': np.mean(upper - lower)
        }
        
        # Calculate win probability metrics
        pred_probs = 1 / (1 + np.exp(-predictions / 100))  # Convert to win probability
        actual_wins = (y_test > 0).astype(int)
        
        metrics['win_probability_calibration'] = np.mean(np.abs(pred_probs - actual_wins))
        
        return metrics 