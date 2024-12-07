"""
EDP Analysis package for NFL game prediction.
"""

from .edp_calculator import EDPCalculator
from .opponent_strength import OpponentStrengthAdjuster
from .game_prediction import EDPGamePredictor

__all__ = [
    'EDPCalculator',
    'OpponentStrengthAdjuster',
    'EDPGamePredictor'
]
