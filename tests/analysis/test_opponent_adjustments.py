"""
Tests for opponent strength adjustments in EDP calculations.
"""

import pandas as pd
import numpy as np
import pytest
from analysis.edp_calculation.opponent_strength import (
    calculate_opponent_strength_metrics,
    calculate_matchup_adjustments,
    apply_opponent_adjustments
)
from processing.core.config import OPPONENT_STRENGTH_SETTINGS

@pytest.fixture
def sample_team_data():
    """Create sample team performance data for testing."""
    return pd.DataFrame({
        'game_id': range(1, 11),
        'team': ['KC', 'KC', 'KC', 'KC', 'KC', 'BUF', 'BUF', 'BUF', 'BUF', 'BUF'],
        'weighted_offensive_edp': [0.5, 0.6, 0.4, 0.7, 0.5, 0.3, 0.4, 0.5, 0.4, 0.6],
        'weighted_defensive_edp': [0.3, 0.4, 0.2, 0.5, 0.4, 0.2, 0.3, 0.4, 0.3, 0.5]
    })

def test_opponent_strength_calculation(sample_team_data):
    """Test that opponent strength metrics are calculated correctly."""
    strength_metrics = calculate_opponent_strength_metrics(
        sample_team_data,
        metrics=OPPONENT_STRENGTH_SETTINGS['metrics'],
        window=OPPONENT_STRENGTH_SETTINGS['window']
    )
    
    assert not strength_metrics.empty
    assert 'opponent_weighted_offensive_edp' in strength_metrics.columns
    assert 'opponent_weighted_defensive_edp' in strength_metrics.columns
    assert len(strength_metrics) == len(sample_team_data)

def test_matchup_adjustments(sample_team_data):
    """Test that matchup adjustments are calculated within expected bounds."""
    opponent_metrics = calculate_opponent_strength_metrics(
        sample_team_data,
        metrics=OPPONENT_STRENGTH_SETTINGS['metrics']
    )
    
    adjustments = calculate_matchup_adjustments(
        sample_team_data[['weighted_offensive_edp', 'weighted_defensive_edp']],
        opponent_metrics,
        weights=OPPONENT_STRENGTH_SETTINGS['weights']
    )
    
    assert not adjustments.empty
    assert 'matchup_adjustment' in adjustments.columns
    # Check that adjustments are within configured bounds
    assert adjustments['matchup_adjustment'].abs().max() <= OPPONENT_STRENGTH_SETTINGS['max_adjustment']

def test_edp_adjustments(sample_team_data):
    """Test that EDP adjustments are applied correctly."""
    opponent_metrics = calculate_opponent_strength_metrics(
        sample_team_data,
        metrics=OPPONENT_STRENGTH_SETTINGS['metrics']
    )
    
    adjustments = calculate_matchup_adjustments(
        sample_team_data[['weighted_offensive_edp', 'weighted_defensive_edp']],
        opponent_metrics
    )
    
    adjusted_edp = apply_opponent_adjustments(
        sample_team_data,
        'weighted_offensive_edp',
        opponent_metrics,
        adjustments
    )
    
    assert len(adjusted_edp) == len(sample_team_data)
    # Check that adjustments maintain reasonable ranges
    assert (adjusted_edp.abs() <= sample_team_data['weighted_offensive_edp'].abs() * 1.5).all()

def test_adjustment_consistency(sample_team_data):
    """Test that adjustments are consistent across similar situations."""
    # Create duplicate data with same metrics
    duplicate_data = sample_team_data.copy()
    duplicate_data['game_id'] = duplicate_data['game_id'] + 100
    
    combined_data = pd.concat([sample_team_data, duplicate_data])
    
    opponent_metrics = calculate_opponent_strength_metrics(
        combined_data,
        metrics=OPPONENT_STRENGTH_SETTINGS['metrics']
    )
    
    adjustments = calculate_matchup_adjustments(
        combined_data[['weighted_offensive_edp', 'weighted_defensive_edp']],
        opponent_metrics
    )
    
    # Check that identical situations get identical adjustments
    first_half = adjustments.iloc[:len(sample_team_data)]
    second_half = adjustments.iloc[len(sample_team_data):]
    
    pd.testing.assert_series_equal(
        first_half['matchup_adjustment'],
        second_half['matchup_adjustment'],
        check_names=False
    ) 