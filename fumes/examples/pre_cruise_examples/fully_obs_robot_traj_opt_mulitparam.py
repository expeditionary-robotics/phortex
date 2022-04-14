"""Joins a spatiotemporal world with a lawnmower trajectory.

Optimizes over lawnmower height, width, orientation, and origin, subject
to a trajectory length budget constraint.

# TODO: for some reason, optimizing lawnmower resolution does not work -
the optimizers fail to converge.


Example usage (within the fumes environment):
    python fumes/examples/fully_obs_robot_traj_opt_multiparam.py
"""

import numpy as np
import matplotlib.pyplot as plt

from fumes.environment import Bullseye
from fumes.environment.utils import xcoord_circle, ycoord_circle

from fumes.trajectory import Lawnmower, Spiral
from fumes.planner import TrajectoryOpt, LawnSpiralGeneratorFlexible

from fumes.model import FullyObs
from fumes.robot import OfflineRobot
from fumes.simulator import Simulator

from fumes.reward import SampleValues
from fumes.environment import Extent

# Parameters
experiment_name = "traj_opt_size"

# Trajectory params
traj_type = "lawnmower"  # type of fixed trajectory
resolution = 10  # lawnmower resolution (in meters)

# Reward params
thresh = 0.5  # in-plume threshhold, in concentration units

# Environment params
nx = 500  # environment x width (in meters)
ny = 500  # environment y height (in meters)
extent = Extent(xrange=(0., 500.), xres=100, yrange=(0., 500.), yres=100)

# Simulation params
t0 = 0.0  # initial time
duration = 12  # duration in hours

# Robot params
vel = 0.5  # robot velocity (in meters/second)
samp_dist = 0.5  # distance between samples (in meters)
com_window = 120  # communication window (in seconds)

# Create the environment
environment = Bullseye(nx, ny, nx, ny, xcoord_circle, ycoord_circle,
                       l=0.01, A=1.0)

# Create a fully observeable model
model = FullyObs(extent, environment)

# Create the base trajectory generator object
traj_generator = LawnSpiralGeneratorFlexible(t0, vel, traj_type, resolution)

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)

# Create planner
# x0 is of form (lh, lw, rotation, origin_x, origin_y)
planner = TrajectoryOpt(
    model,
    traj_generator.generate,
    reward,
    x0=(100., 500., 135., 200., 200.),
    budget=3000.,
    limits=[extent.xmin, extent.xmax, extent.ymin, extent.ymax]
    # limits=None,
)
plan_opt = planner.get_plan()

# Create the robot
rob = OfflineRobot(model, plan_opt, environment, vel, com_window)

# Create the simulator
simulator = Simulator(rob, environment)

# Run the simulator
times = np.linspace(0, duration * 60 * 60, duration * 60 * 60 + 1)
simulator.simulate(times, experiment_name=experiment_name)

# Plot outcomes
simulator.plot_comms()
simulator.plot_all()
simulator.plot_world()
