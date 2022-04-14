"""Joins a spatiotemporal world with a fixed lawnmower trajectory."""

import numpy as np
import distfit
import scipy as sp
import pdb

from fumes.environment.mtt import StationaryMTT
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S
from fumes.model.mtt import MTT, Parameter
from fumes.trajectory import Lawnmower, Spiral
from fumes.planner import TrajectoryOpt
from fumes.simulator import Simulator
from fumes.robot import OfflineRobot
from fumes.model import FullyObs
from fumes.reward import SampleValues


# Parameters
experiment_name = "mtt_traj_opt"

# Trajectory params
traj_type = "lawnmower"  # type of fixed trajectory
lh = 250  # lawnmower height (in meters)
lw = 250  # lawnmower width (in meters)
origin = (10., 10.)  # starting point
resolution = 5  # lawnmower resolution (in meters)
altitude = 100  # altitude where trajectory is located (in meters)

# Environment params
nx = 500  # environment x width (in meters)
ny = 500  # environment y height (in meters)

# Simulation params
t0 = 0.0
duration = 12 * 3600.  # duration (in seconds)

# Robot params
vel = 0.5  # robot velocity (in meters/second)
com_window = 120  # communication window (in seconds)
samp_dist = 0.5  # distance between samples (in meters)


# Create the environment
z = np.linspace(0, 200, 100)  # height to integrate over
loc = (0, 0)  # coordinates easting, northing
tprof = pacific_sp_T  # function that describes background temp
sprof = pacific_sp_S  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T
v0 = 0.4  # source exit velocity
a0 = 0.1  # source area
s0 = 34.608  # source salinity
t0 = 300  # source temperature
rho0 = eos_rho(t0, s0)  # source density

# Inferred Source Params
v0_inf = distfit(distr='uniform')
v0_inf.fit_transform(np.random.uniform(0.05, 1.5, 2000))
v0_prop = sp.stats.norm(loc=0, scale=1.0)
v0_param = Parameter(v0_inf, v0_prop)

a0_inf = distfit(distr='uniform')
a0_inf.fit_transform(np.random.uniform(0.05, 0.5, 2000))
a0_prop = sp.stats.norm(loc=0, scale=0.1)
a0_param = Parameter(a0_inf, a0_prop)

E_inf = distfit(distr='uniform')
E_inf.fit_transform(np.random.uniform(0.1, 0.4, 2000))
E_prop = sp.stats.norm(loc=0, scale=0.1)
E_param = Parameter(E_inf, E_prop)

# Simulation Params
xdim = 200  # target simulation xdim
ydim = 200  # target simulation ydim
zdim = 200  # target simulatino zdim
resolution = 10  # number of voxels in each dimension
iter = 100  # number of samples to search over
burn = 50  # number of samples for burn-in
thresh = 1e-5  # probability threshold for a detection

environment = StationaryMTT(plume_loc=loc, z=z, tprof=tprof, sprof=sprof,
                            rhoprof=rhoprof, vex=v0, area=a0, density=rho0,
                            salt=s0, temp=t0)

# Create Model
model = MTT(plume_loc=(0, 0), z=z, tprof=tprof, sprof=sprof, rhoprof=rhoprof,
            vex=v0_param, area=a0_param, density=rho0, salt=s0, temp=t0,
            E=E_param)

# Create the base trajectory generator object


def traj_generator(origin_x=0.0, origin_y=0.0, orientation=0.):
    """Returns a trajectory object"""
    if traj_type == "lawnmower":
        return Lawnmower(t0,
                         vel,
                         lh,
                         lw,
                         resolution,
                         altitude=altitude,
                         noise=0.0,
                         orientation=orientation,
                         origin=(origin_x, origin_y))
    elif traj_type == "spiral":
        return Spiral(t0,
                      vel,
                      lh,
                      lw,
                      resolution,
                      altitude=altitude,
                      noise=0.0,
                      orientation=orientation,
                      origin=(origin_x, origin_y))
    else:
        raise ValueError(f"Unrecognized trajectory type {traj_type}.")


# Uniform sampling params
sampling_params = {
    "samp_dist": samp_dist
}

# Reward function
reward = SampleValues(sampling_params=sampling_params, is_cost=True)
planner = TrajectoryOpt(model, traj_generator, reward, x0=(10.0, 10.0, 0.0))
plan_opt = planner.get_plan()

# Create the robot
rob = OfflineRobot(model, plan_opt, environment, vel, com_window)

# Create the simulator
simulator = Simulator(rob, environment)

# Run the simulator
times = np.linspace(0, 12 * 60 * 60, 12 * 60 * 60 + 1)
simulator.simulate(times, experiment_name=experiment_name)

# Plot outcomes
simulator.plot_comms()
simulator.plot_all()

# Update model
obs = [float(o[-1] > 1e-5) for o in simulator.obs]
_, updateE, updateV = model.update_multiple_class(times, simulator.coords, obs)
print(updateE, updateV)
