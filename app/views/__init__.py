"""
Pages package for Streamlit app
Each module represents one app page
"""

from . import (
    home,
    dataset,
    model,
    training,
    monitor,
    results,
    interpretability
)

__all__ = [
    'home',
    'dataset',
    'model',
    'training',
    'monitor',
    'results',
    'interpretability'
]
