"""
Unit tests for EDPCalculator V2.
"""

import unittest
import pandas as pd
import numpy as np
from analysis.edp_calculation.edp_calculator_v2 import EDPCalculator, EDPWeights
from processing.pbp.pipeline import NFLDataPipeline


class TestEDPCalculatorV2(unittest.TestCase):
    """Test cases for EDPCalculator V2."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that should be reused across all tests."""
        # Load real NFL data
        cls.pipeline = NFLDataPipeline(seasons=[2024])
        cls.play_data = cls.pipeline.run_pipeline()
        
        if cls.play_data.empty:
            raise ValueError("Failed to load play-by-play data")

    def setUp(self):
        """Set up test fixtures."""
        self.calculator = EDPCalculator()
        
        # Extract a single drive for focused testing
        self.single_drive = self.play_data[
            (self.play_data['game_id'] == self.play_data['game_id'].iloc[0]) &
            (self.play_data['drive'] == self.play_data['drive'].iloc[0])
        ].copy()

    def test_preprocess_data(self):
        """Test data preprocessing."""
        processed_df = self.calculator.preprocess_data(self.single_drive)
        
        # Check required columns exist
        required_cols = ['drive_num', 'week', 'season']
        for col in required_cols:
            self.assertTrue(col in processed_df.columns, f"Missing column: {col}")
        
        # Check drive_num format
        drive_num = processed_df['drive_num'].iloc[0]
        self.assertTrue('_' in drive_num, "drive_num should contain underscores")
        
        # Verify no null values in key columns
        key_cols = ['week', 'season', 'drive_num']
        for col in key_cols:
            self.assertFalse(processed_df[col].isnull().any(),
                           f"Found null values in {col}")

    def test_probability_points(self):
        """Test probability points calculation."""
        prob_points = self.calculator.calculate_probability_points(self.single_drive)
        
        # Ensure probability points are within expected range
        self.assertGreaterEqual(prob_points, 0)
        self.assertLessEqual(prob_points, self.calculator.weights.PROB_POINTS_WEIGHT)
        
        # Test with empty drive
        empty_drive = self.single_drive.iloc[0:0]
        empty_points = self.calculator.calculate_probability_points(empty_drive)
        self.assertEqual(empty_points, 0)

    def test_volatility_penalty(self):
        """Test volatility penalty calculation."""
        volatility = self.calculator.calculate_volatility_penalty(self.single_drive)
        
        # Ensure volatility penalty is negative or zero
        self.assertLessEqual(volatility, 0)
        
        # Test with consistent gains
        consistent_drive = self.single_drive.copy()
        consistent_drive['yards_gained'] = 5
        consistent_volatility = self.calculator.calculate_volatility_penalty(consistent_drive)
        
        # Volatility should be lower (closer to 0) for consistent gains
        self.assertGreater(consistent_volatility, volatility)

    def test_drive_quality(self):
        """Test overall drive quality calculation."""
        drive_metrics = self.calculator.calculate_drive_quality(self.play_data)
        
        # Check required columns
        required_cols = ['drive_num', 'earned_drive_points']
        for col in required_cols:
            self.assertTrue(col in drive_metrics.columns)
        
        # Check value ranges
        self.assertTrue(all(drive_metrics['earned_drive_points'] >= 0))
        max_possible = (self.calculator.weights.PROB_POINTS_WEIGHT +
                       self.calculator.weights.EXPLOSIVE_PLAY_WEIGHT * 3)  # Assume max 3 explosive plays
        self.assertTrue(all(drive_metrics['earned_drive_points'] <= max_possible))

    def test_team_metrics(self):
        """Test team-level metrics calculation."""
        offensive_stats, defensive_stats = self.calculator.calculate_team_metrics(
            self.play_data
        )
        
        # Check team counts
        self.assertEqual(len(offensive_stats), 32)  # All NFL teams
        self.assertEqual(len(defensive_stats), 32)
        
        # Check required columns
        off_cols = ['team', 'edp_per_drive_off', 'total_drives_off']
        def_cols = ['team', 'edp_per_drive_def', 'total_drives_def']
        
        for col in off_cols:
            self.assertTrue(col in offensive_stats.columns)
        for col in def_cols:
            self.assertTrue(col in defensive_stats.columns)
        
        # Check value ranges
        self.assertTrue(all(offensive_stats['edp_per_drive_off'] >= 0))
        self.assertTrue(all(defensive_stats['edp_per_drive_def'] >= 0))
        
        # Check drive counts
        self.assertTrue(all(offensive_stats['total_drives_off'] > 0))
        self.assertTrue(all(defensive_stats['total_drives_def'] > 0))


if __name__ == '__main__':
    unittest.main() 