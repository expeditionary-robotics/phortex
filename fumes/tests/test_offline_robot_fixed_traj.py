"""Joins a spatiotemporal world with a fixed lawnmower trajectory."""

import numpy as np
import pdb

from fumes.environment import Bullseye
from fumes.environment.utils import xcoord_circle, ycoord_circle
from fumes.trajectory import Lawnmower, Spiral
from fumes.planner import FixedTrajectory
from fumes.simulator import Simulator
from fumes.robot import OfflineRobot
from fumes.model import FullyObs

def test_fixed_trajectory_offline_planner():
    # Parameters
    traj_type = "lawnmower" # type of fixed trajectory
    lh = 500  # lawnmower height (in meters)
    lw = 500  # lawnmower width (in meters)
    resolution = 10  # lawnmower resolution (in meters)
    nx = 500  # environment x width (in meters)
    ny = 500  # environment y height (in meters)
    vel = 0.5 # robot velocity (in meters/second)
    com_window = 120 # communication window (in seconds)
    t0 = 0.0

    # Create the planner
    if traj_type == "lawnmower":
        trajectory_base = Lawnmower(t0, vel, lh, lw, resolution, noise=0.0)
    elif traj_type == "spiral":
        trajectory_base = Spiral(t0, vel, lh, lw, resolution, noise=0.0)
    else:
        raise ValueError(f"Unknown trajectory type {traj_type}.")

    planner = FixedTrajectory(trajectory_base)
    trajectory = planner.get_plan()

    # Create the environment
    environment = Bullseye(nx, ny, nx, ny, xcoord_circle, ycoord_circle,
                        l=0.01, A=1.0)

    # Create a fully observeable model
    model = FullyObs(environment)

    # Create the robot
    rob = OfflineRobot(model, trajectory, environment, vel, com_window)

    # Create the simulator
    simulator = Simulator(rob, environment)

    # Run the simulator
    times = np.linspace(0, 12*60*60, 12*60*60+1)
    simulator.simulate(times)

    # Plot outcomes
    simulator.plot_comms()
    simulator.plot_all()
    simulator.plot_world()
