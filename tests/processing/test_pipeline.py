"""
Unit tests for the NFL EDP analysis pipeline.
"""

from typing import Dict, Optional, Any
import logging
import pandas as pd
import traceback
from data_pipeline import NFLDataPipeline
from utils.logging_config import setup_logging
from utils.data_validation import DataValidator

logger = setup_logging(logging.INFO, module_name=__name__)
validator = DataValidator()

def test_pipeline() -> Optional[pd.DataFrame]:
    """
    Test the basic pipeline functionality.
    
    Returns:
        Optional[pd.DataFrame]: Raw data if tests pass, None if any test fails.
    """
    try:
        # Initialize pipeline with current season
        pipeline = NFLDataPipeline(seasons=[2023], log_level=logging.INFO)
        
        # Test raw data loading with validation
        logger.info("1. Testing raw data loading...")
        raw_data = pipeline.load_raw_data()
        
        # Validate raw data
        required_cols = ['play_id', 'game_id', 'posteam', 'defteam']
        is_valid, results = validator.validate_dataset_completeness(
            raw_data, required_cols, "raw_data"
        )
        
        if not is_valid:
            logger.error("Raw data validation failed")
            logger.error(f"Validation results: {results}")
            return None
            
        # Test team name standardization
        team_cols = ['home_team', 'away_team', 'posteam', 'defteam']
        if not validator.validate_team_names(raw_data, team_cols):
            logger.error("Team name validation failed")
            return None
        
        return raw_data
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None

def test_full_pipeline() -> Optional[Dict[str, Any]]:
    """
    Comprehensive test of the entire pipeline.
    
    Returns:
        Optional[Dict[str, Any]]: Dictionary with test results if successful, None if failed.
    """
    try:
        pipeline = NFLDataPipeline(seasons=[2023])
        
        print("\n1. Loading data...")
        df = pipeline.run_pipeline()
        
        if df.empty:
            print("\nPipeline returned empty DataFrame")
            return None
        
        # Get counts for validation
        admin_plays = df['is_administrative_play'].sum()
        timeout_plays = (df['timeout'] == 1).sum()
        remaining_nulls = df['posteam'].isnull().sum()
        
        # Check if remaining nulls match administrative plays with null values
        admin_nulls = df[
            (df['is_administrative_play'] == True) & 
            (df['posteam'].isnull())
        ].shape[0]
        
        print("\nValidation Results:")
        print(f"- Administrative plays: {admin_plays}")
        print(f"- Timeout plays: {timeout_plays}")
        print(f"- Remaining nulls: {remaining_nulls}")
        print(f"- Administrative plays with nulls: {admin_nulls}")
        print(f"- All nulls accounted for: {remaining_nulls == admin_nulls}")
        
        return {
            'df': df,
            'admin_plays': admin_plays,
            'timeout_plays': timeout_plays,
            'remaining_nulls': remaining_nulls,
            'admin_nulls': admin_nulls
        }
        
    except Exception as e:
        print(f"\nPipeline test failed: {str(e)}")
        traceback.print_exc()
        return None

def test_team_names() -> None:
    """Test team name standardization."""
    pipeline = NFLDataPipeline(seasons=[2023])
    df = pipeline.load_raw_data()
    
    # Check unique team counts
    team_cols = ['home_team', 'away_team', 'posteam', 'defteam']
    for col in team_cols:
        unique_teams = df[col].dropna().unique()
        assert len(unique_teams) <= 32, f"Too many unique teams in {col}: {len(unique_teams)}"

def test_scoring_consistency() -> None:
    """Test score calculations and consistency."""
    pipeline = NFLDataPipeline(seasons=[2023])
    df = pipeline.load_raw_data()
    df = pipeline.validate_scoring(df)
    
    # Verify score differentials
    assert (df['score_differential'] == 
           df['posteam_score'] - df['defteam_score']).all(), "Score differential mismatch"

if __name__ == "__main__":
    results = test_full_pipeline()
    if results:
        print("\nPipeline test completed successfully!")