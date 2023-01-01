"""Plot a saved simulation by mission hash.

Example usage:
    python fumes/guaymas_examples/demo2-plot-from-saved-sim.py -m crossflow_gp_cache_14112021-15:08:40
"""
import os
import argparse
import numpy as np
from fumes.utils import tic, toc
from fumes.utils import load_mission


# Create commandline parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mission", action="store",
                    type=str, help="Mission hash to load.",
                    default="crossflow_gp_cache_14112021-15/08/40.")
parse = parser.parse_args()

# Parse commandline input
mission_hash = parse.mission

rob, model, env, simulator = load_mission(mission_hash)

# Observations
n_pts = 20  # number of time points to simulate
times = np.linspace(0, 3600 * 12, 10)
fig_world, fig_global = simulator.plot_world3d(times=times)
fig_world.show()
fig_global.show()
