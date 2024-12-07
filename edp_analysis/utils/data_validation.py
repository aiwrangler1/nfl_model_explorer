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
            'missing_columns': {},
            'null_counts': {},
            'total_rows': len(df)
        }
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results['missing_columns'] = {col: 0 for col in missing_columns}
            self.logger.error(f"Missing columns in {dataset_name}: {missing_columns}")
            return False, validation_results
        
        # Check null counts
        null_counts = df[required_columns].isnull().sum()
        validation_results['null_counts'] = null_counts.to_dict()
        
        # Validate null counts for required columns
        is_valid = True
        for col, null_count in null_counts.items():
            if null_count > 0 and col not in null_allowed_columns:
                is_valid = False
                self.logger.warning(
                    f"Column {col} in {dataset_name} has {null_count} null values "
                    f"({(null_count/len(df))*100:.1f}% of data)"
                )
        
        return is_valid, validation_results

    def validate_team_names(
        self,
        df: pd.DataFrame,
        team_columns: List[str]
    ) -> Tuple[bool, Dict[str, Set[str]]]:
        """
        Validate team name consistency in a dataframe.
        
        Args:
            df (pd.DataFrame): The dataframe to validate.
            team_columns (List[str]): The list of column names containing team names.
            
        Returns:
            Tuple[bool, Dict[str, Set[str]]]: (is_valid, invalid_teams_by_column)
                invalid_teams_by_column maps column names to sets of invalid team names found
        """
        is_valid = True
        invalid_teams_by_column = {}
        
        for col in team_columns:
            if col not in df.columns:
                continue
                
            unique_teams = set(df[col].dropna().unique())
            invalid_teams = unique_teams - self.valid_teams
            
            if invalid_teams:
                is_valid = False
                invalid_teams_by_column[col] = invalid_teams
                self.logger.warning(
                    f"Found invalid team names in {col}: {invalid_teams}"
                )
                
        return is_valid, invalid_teams_by_column

    def validate_numerical_ranges(
        self,
        df: pd.DataFrame,
        range_checks: Dict[str, Tuple[float, float]]
    ) -> Tuple[bool, Dict[str, Dict[str, int]]]:
        """
        Validate that numerical columns are within expected ranges.
        
        Args:
            df (pd.DataFrame): The dataframe to validate.
            range_checks (Dict[str, Tuple[float, float]]): A dictionary mapping column names to (min, max) tuples.
            
        Returns:
            Tuple[bool, Dict[str, Dict[str, int]]]: (is_valid, validation_results)
                validation_results contains for each column:
                - below_min: count of values below minimum
                - above_max: count of values above maximum
                - null_count: count of null values
        """
        is_valid = True
        validation_results = {}
        
        for col, (min_val, max_val) in range_checks.items():
            if col not in df.columns:
                continue
                
            col_results = {
                'below_min': 0,
                'above_max': 0,
                'null_count': df[col].isnull().sum()
            }
            
            # Check ranges for non-null values
            non_null_data = df[col].dropna()
            col_results['below_min'] = (non_null_data < min_val).sum()
            col_results['above_max'] = (non_null_data > max_val).sum()
            
            if col_results['below_min'] > 0 or col_results['above_max'] > 0:
                is_valid = False
                self.logger.warning(
                    f"Column {col} has {col_results['below_min']} values below {min_val} "
                    f"and {col_results['above_max']} values above {max_val}"
                )
            
            validation_results[col] = col_results
                    
        return is_valid, validation_results