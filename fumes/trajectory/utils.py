"""Utility functions for trajectory objects."""
import numpy as np

def distance(x1, y1, x2, y2):
    """Computes Euclidean distance from two points."""
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


