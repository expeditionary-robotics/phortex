import os
import argparse
import numpy as np
from fumes.utils import tic, toc
from fumes.utils import load_mission


# Create commandline parser
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mission", action="store",
                    type=str, help="Mission hash to load.",
                    default="mttcrossflow_07112021-22:48:00")
parse = parser.parse_args()

# Parse commandline input
mission_hash = parse.mission

rob, model, env, simulator = load_mission(mission_hash)
# Plot outcomes
# simulator.plot_comms()
# simulator.plot_all()

# Observations
n_pts = 20  # number of time points to simulate
times = np.linspace(0, 3600 * 12, 10)
fig_world, fig_global = simulator.plot_world3d(times=times)
fig_world.show()
fig_global.show()
