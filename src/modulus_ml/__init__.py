"""
Modulus ML - A comprehensive machine learning model comparison library.
"""

from .comparator import ModelComparator
from .metrics import calculate_metrics
from .visualizer import plot_comparison

__version__ = "0.1.0"

__all__ = ["ModelComparator", "calculate_metrics", "plot_comparison"]