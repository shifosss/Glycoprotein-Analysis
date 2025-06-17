"""
Network module for glycan-protein binding prediction models.

This module provides neural network architectures and prediction utilities
for glycan-protein binding strength prediction tasks.
"""

# Import main components from submodules
from .binding_strength_networks import *
from .binding_strength_predictor import *
from .binding_strength_predictor_v2 import *

# Define what should be imported with "from network import *"
__all__ = [
    # Add your network classes here
    # Example: 'BindingPredictor', 'MLPNetwork'
    'MLPRegressor',
    'ResidualMLPRegressor',
    'ResidualBlock',
    'AttentionRegressor',
    'EnsembleRegressor',
    'EnhancedBindingStrengthPredictor'
]
