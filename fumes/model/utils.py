"""Utilities for model files."""
import numpy as np
import scipy as sp
from scipy import interpolate
import json
from distfit import distfit
import gpytorch as gpy

from scipy.stats import norm
import matplotlib.pyplot as plt


def mcmc_mh(init, gx, px, iter=1000, burnin=100):
    """Metropolis-Hastings Algorithm.

    Args:
        init (float) initial guess
        gx (function) distribution to draw guesses
        px (function) pdf to compute goodness of guess
        iter (int) number of samples to draw
        burnin (int) number of samples allocated to burn in

    Returns:
        states (list[floats]) accepted states
    """
    xt = init
    states = []
    for i in range(iter):
        # Generate random candidate from gx
        x = gx(size=1)[0]
        # Compute acceptance probability
        curr_prob = px(xt)
        move_prob = px(x)
        accept = min(1., move_prob / curr_prob)
        # Determine whether to accept
        if np.random.uniform(size=1) <= accept:
            xt = x
        states.append(xt)
    return states[burnin:]


def normalize_data(target, minmax):
    """Performs target normalization."""
    return np.asarray((target - minmax[0]) / (minmax[1] - minmax[0]))


def unnormalize_data(target, minmax):
    """Performs target normalization."""
    return np.asarray(target * (minmax[1] - minmax[0]) + minmax[0])

