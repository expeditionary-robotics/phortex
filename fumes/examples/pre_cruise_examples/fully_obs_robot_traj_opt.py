"""Joins a spatiotemporal world with a fixed lawnmower trajectory.

Example parameters can be modified at the start of the file.

Example usage (within the fumes environment):
    python fumes/examples/fully_obs_robot_traj_opt.py
"""

import numpy as np
import matplotlib.pyplot as plt

from fumes.environment import Bullseye
from fumes.environment.utils import xcoord_circle, ycoord_circle

from fumes.trajectory import Lawnmower, Spiral
from fumes.planner import TrajectoryOpt, LawnSpiralGenerator

from fumes.model import FullyObs
from fumes.robot import OfflineRobot
from fumes.simulator import Simulator

from fumes.reward import SampleValues

import pdb

# Parameters
experiment_name = "traj_opt_spiral"

# Trajectory params
traj_type = "lawnmower"  # type of fixed trajectory
lh = 100  # lawnmower height (in meters)
lw = 100  # lawnmower width (in meters)
resolution = 10  # lawnmower resolution (in meters)

# Reward params
thresh = 0.5  # in-plume threshhold, in concentration units

# Environment params
nx = 500  # environment x width (in meters)
ny = 500  # environment y height (in meters)

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
model = FullyObs(environment)

# Create the base trajectory generator object
traj_generator = LawnSpiralGenerator(t0, vel, traj_type, lh, lw, resolution)

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)

# Create planner
planner = TrajectoryOpt(
    model, traj_generator.generate, reward,
    x0=(0.0, 200.0, 200.0), budget=30000.)
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
