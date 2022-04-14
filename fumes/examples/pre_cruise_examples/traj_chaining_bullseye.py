"""Joins a spatiotemporal world with a chained lawnmower trajectory.

Example parameters can be modified at the start of the file.

Example usage (within the fumes environment):
    python fumes/examples/traj_chaining_bullseye.py
"""

import numpy as np
import matplotlib.pyplot as plt

from fumes.environment import Bullseye
from fumes.environment.utils import xcoord_circle, ycoord_circle

from fumes.trajectory import Lawnmower, Spiral
from fumes.planner import TrajectoryOpt, TrajectoryChain, \
    LawnSpiralGeneratorFlexible

from fumes.model import FullyObs
from fumes.robot import OfflineRobot
from fumes.simulator import Simulator

from fumes.reward import SampleValues

import pdb

# Parameters
experiment_name = "traj_opt_chain_soft_origin5"

# Trajectory params
traj_type = "lawnmower"  # type of fixed trajectory
resolution = 10  # lawnmower resolution (in meters)
time_resolution = 3600  # time resolution (in seconds)

# Environment params
nx = 500  # environment x width (in meters)
ny = 500  # environment y height (in meters)

# Simulation params
t0 = 0.0  # initial time
duration = 12 * 3600.  # duration (in seconds)

# Robot params
vel = 0.5  # robot velocity (in meters/second)
samp_dist = 0.5  # distance between samples (in meters)
com_window = 120  # communication window (in seconds)

# Create the environment
environment = Bullseye(nx, ny, nx, ny, xcoord_circle, ycoord_circle,
                       l=0.01, A=1.0)

# Create a fully observeable model
model = FullyObs(environment)

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)


planners = []
budget = time_resolution * vel  # distance budget per leg
for start_time in np.arange(0, duration, step=time_resolution):
    # Create the base trajectory generator object
    traj_generator = LawnSpiralGeneratorFlexible(
        start_time, vel, traj_type, resolution)

    # Create planner
    planners.append(TrajectoryOpt(
        model,
        traj_generator.generate,
        reward,
        x0=(200., 200., 0., 200., 200.),  # (lh, lw, rot, origin_x, origin_y)
        budget=budget,
        limits=[0., 500., 0., 500.]
    ))

planner = TrajectoryChain(planners=planners)
plan_opt = planner.get_plan()

# Create the robot
rob = OfflineRobot(model, plan_opt, environment, vel, com_window)

# Create the simulator
simulator = Simulator(rob, environment)

# Run the simulator
times = np.arange(0, duration + 1)
simulator.simulate(times, experiment_name=experiment_name)

# Plot outcomes
simulator.plot_comms()
simulator.plot_all()
simulator.plot_world()
