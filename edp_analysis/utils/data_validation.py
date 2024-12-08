"""
Utility functions for validating NFL data.
"""

import pandas as pd
import logging
from typing import List, Dict, Tuple, Optional, Set

class DataValidator:
    """
    A class to encapsulate various data validation functions.
    """
    
    def __init__(self):
        """Initialize validator with known valid team names and aliases."""
        self.logger = logging.getLogger(__name__)
        # Include historical and current team names
        self.valid_teams: Set[str] = {
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE',
            'DAL', 'DEN', 'DET', 'GB', 'HOU', 'IND', 'JAX', 'JAC',
            'KC', 'LAC', 'LAR', 'LV', 'MIA', 'MIN', 'NE', 'NO',
            'NYG', 'NYJ', 'OAK', 'PHI', 'PIT', 'SEA', 'SF', 'TB',
            'TEN', 'WAS', 'WSH', 'STL', 'SD', 'LA'
        }
    
    def validate_dataset_completeness(
        self,
        df: pd.DataFrame, 
        required_columns: List[str],
        dataset_name: str,
        null_allowed_columns: Optional[List[str]] = None
    ) -> Tuple[bool, Dict[str, Dict[str, int]]]:
        """
        Validate that a dataset has all required columns and acceptable null counts.
        
        Args:
            df (pd.DataFrame): The dataframe to validate.
            required_columns (List[str]): The list of required column names.
            dataset_name (str): The name of the dataset being validated.
            null_allowed_columns (Optional[List[str]]): Columns where nulls are acceptable.
            
        Returns:
            Tuple[bool, Dict[str, Dict[str, int]]]: A tuple of (is_valid, validation_results).
                validation_results contains:
                - missing_columns: columns not present in the dataframe
                - null_counts: number of null values per column
                - total_rows: total number of rows in the dataset
        """
        null_allowed_columns = null_allowed_columns or []
        validation_results = {
            'missing_columns': [],
            'null_counts': {},
            'total_rows': len(df)
        }
        
        # Check for missing columns
        missing_cols = [col for col in required_columns if col not in df.columns]
        validation_results['missing_columns'] = missing_cols
        
        if missing_cols:
            self.logger.warning(f"Missing required columns in {dataset_name}: {missing_cols}")
            return False, validation_results
        
        # Check for null values
        null_counts = df[required_columns].isnull().sum()
        validation_results['null_counts'] = null_counts.to_dict()
        
        # Determine if dataset is valid based on null counts
        is_valid = True
        for col, null_count in null_counts.items():
            if null_count > 0 and col not in null_allowed_columns:
                self.logger.warning(
                    f"Column {col} in {dataset_name} has {null_count} null values"
                )
                is_valid = False
        
        return is_valid, validation_results
    
    def validate_team_names(
        self,
        df: pd.DataFrame,
        team_columns: List[str]
    ) -> Dict[str, any]:
        """
        Validate team names in specified columns against known valid teams.
        
        Args:
            df (pd.DataFrame): The dataframe containing team names
            team_columns (List[str]): List of columns containing team names
            
        Returns:
            Dict containing validation results:
                - is_valid: bool indicating if all team names are valid
                - invalid_teams: dict mapping columns to sets of invalid team names
        """
        results = {
            'is_valid': True,
            'invalid_teams': {}
        }
        
        for col in team_columns:
            if col not in df.columns:
                self.logger.warning(f"Team column {col} not found in dataframe")
                continue
                
            unique_teams = set(df[col].dropna().unique())
            invalid_teams = unique_teams - self.valid_teams
            
            if invalid_teams:
                results['is_valid'] = False
                results['invalid_teams'][col] = invalid_teams
                self.logger.warning(f"Invalid team names found in {col}: {invalid_teams}")
        
        return results
    
    def validate_numerical_ranges(
        self,
        df: pd.DataFrame,
        column_ranges: Dict[str, Tuple[float, float]]
    ) -> Dict[str, any]:
        """
        Validate that numerical columns fall within expected ranges.
        
        Args:
            df (pd.DataFrame): The dataframe to validate
            column_ranges (Dict[str, Tuple[float, float]]): Dict mapping column names to (min, max) ranges
            
        Returns:
            Dict containing validation results:
                - is_valid: bool indicating if all values are within ranges
                - out_of_range: dict mapping columns to counts of out-of-range values
        """
        results = {
            'is_valid': True,
            'out_of_range': {}
        }
        
        for col, (min_val, max_val) in column_ranges.items():
            if col not in df.columns:
                self.logger.warning(f"Column {col} not found in dataframe")
                continue
                
            # Count values outside the expected range
            out_of_range = df[
                (df[col] < min_val) | (df[col] > max_val)
            ].shape[0]
            
            if out_of_range > 0:
                results['is_valid'] = False
                results['out_of_range'][col] = out_of_range
                self.logger.warning(
                    f"Column {col} has {out_of_range} values outside range [{min_val}, {max_val}]"
                )
        
        return results