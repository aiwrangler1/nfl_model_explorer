"""
Configuration Settings

This module contains configuration constants and settings for the EDP analysis package.
"""

from typing import Dict, Set, List
from dataclasses import dataclass, field
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
MODEL_OUTPUTS_DIR = PROJECT_ROOT / 'model_outputs'

@dataclass
class DataConfig:
    """Configuration for data loading and validation."""
    # Required columns for play-by-play data
    REQUIRED_COLUMNS: Set[str] = field(default_factory=lambda: {
        'game_id', 'play_id', 'posteam', 'defteam', 'week', 'season',
        'drive', 'down', 'yards_gained', 'play_type', 'yardline_100',
        'ydstogo', 'ep', 'epa'
    })
    
    # Columns that can have null values
    NULLABLE_COLUMNS: Set[str] = field(default_factory=lambda: {
        'ep', 'epa', 'yards_gained'
    })
    
    # Valid play types for analysis
    VALID_PLAY_TYPES: Set[str] = field(default_factory=lambda: {
        'pass', 'run', 'punt', 'field_goal', 'extra_point',
        'kickoff', 'penalty', 'no_play', 'qb_kneel', 'qb_spike'
    })
    
    # Team name standardization mappings
    TEAM_NAME_MAP: Dict[str, str] = field(default_factory=lambda: {
        # Current teams (as of 2024)
        'ARI': 'ARI',  # Arizona Cardinals
        'ATL': 'ATL',  # Atlanta Falcons
        'BAL': 'BAL',  # Baltimore Ravens
        'BUF': 'BUF',  # Buffalo Bills
        'CAR': 'CAR',  # Carolina Panthers
        'CHI': 'CHI',  # Chicago Bears
        'CIN': 'CIN',  # Cincinnati Bengals
        'CLE': 'CLE',  # Cleveland Browns
        'DAL': 'DAL',  # Dallas Cowboys
        'DEN': 'DEN',  # Denver Broncos
        'DET': 'DET',  # Detroit Lions
        'GB': 'GB',    # Green Bay Packers
        'HOU': 'HOU',  # Houston Texans
        'IND': 'IND',  # Indianapolis Colts
        'JAX': 'JAC',  # Jacksonville Jaguars
        'JAG': 'JAC',  # Jacksonville alternate
        'KC': 'KC',    # Kansas City Chiefs
        'LAC': 'LAC',  # Los Angeles Chargers
        'LAR': 'LAR',  # Los Angeles Rams
        'LV': 'LV',    # Las Vegas Raiders
        'MIA': 'MIA',  # Miami Dolphins
        'MIN': 'MIN',  # Minnesota Vikings
        'NE': 'NE',    # New England Patriots
        'NO': 'NO',    # New Orleans Saints
        'NYG': 'NYG',  # New York Giants
        'NYJ': 'NYJ',  # New York Jets
        'PHI': 'PHI',  # Philadelphia Eagles
        'PIT': 'PIT',  # Pittsburgh Steelers
        'SEA': 'SEA',  # Seattle Seahawks
        'SF': 'SF',    # San Francisco 49ers
        'TB': 'TB',    # Tampa Bay Buccaneers
        'TEN': 'TEN',  # Tennessee Titans
        'WAS': 'WAS',  # Washington Commanders
        
        # Historical mappings
        'OAK': 'LV',   # Oakland -> Las Vegas Raiders
        'SD': 'LAC',   # San Diego -> LA Chargers
        'STL': 'LAR',  # St. Louis -> LA Rams
    })
    
    # Expected data volume checks
    EXPECTED_GAMES_PER_SEASON: int = 32 * 17  # 32 teams, 17 games each
    MIN_PLAYS_PER_GAME: int = 60
    MAX_PLAYS_PER_GAME: int = 200
    
    # Numeric range validations
    NUMERIC_RANGES: Dict[str, tuple] = field(default_factory=lambda: {
        'yards_gained': (-99, 100),
        'yardline_100': (1, 99),
        'down': (1, 4),
        'ydstogo': (0, 50),
        'ep': (-7, 7),
        'epa': (-15, 15)
    })

    # File patterns and formats
    DATA_FILE_PATTERN: str = "pbp_*.parquet"
    CACHE_FILE_PATTERN: str = "processed_pbp_*.parquet"


@dataclass
class EDPConfig:
    """Configuration for EDP calculations."""
    # Component weights for drive quality
    PROB_POINTS_WEIGHT: float = 0.45
    VOLATILITY_WEIGHT: float = 0.225
    FIELD_POS_WEIGHT: float = 0.225
    MOD_YPP_WEIGHT: float = 0.10
    
    # Success rate thresholds
    EARLY_DOWN_SUCCESS_THRESHOLD: float = 0.5
    LATE_DOWN_SUCCESS_THRESHOLD: float = 1.0
    
    # Explosive play definitions
    EXPLOSIVE_RUN_YARDS: int = 12
    EXPLOSIVE_PASS_YARDS: int = 20
    
    # Field position zones (yards from own goal)
    BACKED_UP_ZONE: tuple = (0, 10)
    SCORING_ZONE: tuple = (75, 100)
    
    # Drive quality thresholds
    MIN_PLAYS_PER_DRIVE: int = 1
    MAX_PLAYS_PER_DRIVE: int = 30
    
    # Minimum drives for rankings
    MIN_DRIVES_FOR_RANKING: int = 20


@dataclass
class SoSConfig:
    """Configuration for Strength of Schedule calculations."""
    # Metrics to consider for opponent strength
    METRICS: List[str] = field(default_factory=lambda: ['offensive_edp', 'defensive_edp'])
    
    # Rolling window for opponent strength calculation
    WINDOW_SIZE: int = 5
    
    # Component weights
    WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'offensive_edp': 0.5,
        'defensive_edp': 0.5
    })
    
    # Maximum adjustment factor (+/- 20%)
    MAX_ADJUSTMENT: float = 0.2


@dataclass
class OutputConfig:
    """Configuration for output generation."""
    # Excel output settings
    EXCEL_SHEET_NAMES: Dict[str, str] = field(default_factory=lambda: {
        'season': 'Season Rankings',
        'weekly': 'Week {week}',
        'team': '{team} Details'
    })
    
    # Metrics rounding
    METRICS_TO_ROUND: Dict[str, int] = field(default_factory=lambda: {
        'edp': 3,
        'epa': 3,
        'success_rate': 3,
        'yards_per_play': 2
    })
    
    # Default sort columns
    SORT_COLUMNS: List[str] = field(default_factory=lambda: [
        'total_edp',
        'offensive_edp',
        'defensive_edp'
    ])
    
    # Visualization settings
    PLOT_DPI: int = 300
    PLOT_FIGSIZE: tuple = (12, 8)
    PLOT_STYLE: str = "whitegrid"


# Global configuration instances
DATA_CONFIG = DataConfig()
EDP_CONFIG = EDPConfig()
SOS_CONFIG = SoSConfig()
OUTPUT_CONFIG = OutputConfig()