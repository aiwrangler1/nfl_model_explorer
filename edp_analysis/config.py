"""
Configuration settings for the NFL EDP analysis codebase.
"""

# Team name mappings
TEAM_NAME_MAP = {
    'JAX': 'JAC', 'LA': 'LAR', 'LV': 'LAS', 'SD': 'LAC',
    'STL': 'LAR', 'OAK': 'LAS', 'JAG': 'JAC'
}

# Expected number of games per season
EXPECTED_GAMES_PER_SEASON = 32 * 17

# Minimum roster size threshold
MIN_ROSTER_SIZE = 45

# Key positions for injury metrics
KEY_POSITIONS = ['QB', 'WR', 'RB', 'TE', 'OL']
OFFENSIVE_POSITIONS = ['qb', 'wr', 'rb', 'te', 'ol']

# EDP calculation coefficients
EDP_COEFFICIENTS = {
    'drive_epa': 0.3,
    'drive_success_rate': 10,
    'num_explosive_plays': 2,
    'num_late_down_conversions': 1.5,
    'scoring_zone_successes': 2,
    'red_zone_failure': -1.5
}

# Opponent strength adjustment settings
OPPONENT_STRENGTH_SETTINGS = {
    'metrics': ['offensive_edp', 'defensive_edp'],
    'window': 5,
    'weights': {
        'offensive_edp': 0.4,
        'defensive_edp': 0.4
    },
    'max_adjustment': 0.2  # Maximum adjustment factor (+/- 20%)
}

# Explosive play threshold (yards)
EXPLOSIVE_PLAY_THRESHOLD = 20

# Rolling EDP window sizes
SHORT_EDP_WINDOW = 5
LONG_EDP_WINDOW = 16

# EDP exponential decay factor
EDP_DECAY_FACTOR = 0.8

# Win probability model settings
WP_DECAY_FACTOR = 0.0007
WP_CALIBRATION_BINS = 10

# Logging settings
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Special Teams EPA Weights
ST_EPA_WEIGHTS = {
    'field_goal': 1.0,    # Full weight for FG outcomes
    'punt': 0.7,          # Moderate weight for punt outcomes
    'kickoff': 0.5,       # Lower weight for kickoff outcomes
    'extra_point': 0.3    # Lowest weight for XP outcomes
}