"""Demonstrates trajectory optimization over crossflow environment."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from distfit import distfit
import os

from fumes.model.mtt import Crossflow, Parameter
from fumes.environment.mtt import CrossflowMTT
from fumes.environment import Extent
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.trajectory import Lawnmower, Spiral
from fumes.planner import TrajectoryOpt, TrajectoryChain, \
    LawnSpiralWithStartGeneratorFlexible
from fumes.reward import SampleValues
from fumes.robot import OfflineRobot
from fumes.simulator import Simulator
from fumes.utils import tic, toc
from fumes.utils import save_mission, get_mission_hash, load_mission

# Environment params
extent = Extent(xrange=(-500., 500.), xres=100, yrange=(-500., 500.), yres=100, zrange=(0., 200.), zres=100,
                global_origin=(27.407489, -111.389893, -1848.5))

# "Global" Parameters
s = np.linspace(0, 1000, 200)  # distance to integrate over
tprof = pacific_sp_T  # function that describes background temp
sprof = pacific_sp_S  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T
T = np.linspace(0, 5*3600, 5)  # make a plan for every hour

# True Source Params
v0 = 0.4  # source exit velocity
a0 = 0.1  # source area
s0 = 34.608  # source salinity
t0 = 300  # source temperature
rho0 = eos_rho(t0, s0)  # source density
E = (0.12, 0.2)

# Inferred Source Params
v0_inf = distfit(distr='uniform')
v0_inf.fit_transform(np.random.uniform(0.05, 1.5, 2000))
v0_prop = sp.stats.norm(loc=0, scale=0.1)
v0_param = Parameter(v0_inf, v0_prop)

a0_inf = distfit(distr='uniform')
a0_inf.fit_transform(np.random.uniform(0.05, 0.5, 2000))
a0_prop = sp.stats.norm(loc=0, scale=0.1)
a0_param = Parameter(a0_inf, a0_prop)

alph_inf = distfit(distr='uniform')
alph_inf.fit_transform(np.random.uniform(0.1, 0.2, 2000))
alph_prop = sp.stats.norm(loc=0, scale=0.01)
alph_param = Parameter(alph_inf, alph_prop)

bet_inf = distfit(distr='uniform')
bet_inf.fit_transform(np.random.uniform(0.05, 0.25, 2000))
bet_prop = sp.stats.norm(loc=0, scale=0.05)
bet_param = Parameter(bet_inf, bet_prop)

# Parameter Optimization Params
iterations = 100  # number of samples to search over
burn = 20  # number of samples for burn-in
thresh = 1e-15  # probability threshold for a detection
# Trajectory Params
traj_type = "lawnmower"  # type of fixed trajectory
traj_res = 5  # lawnmower resolution (in meters)
time_resolution = 3600  # time resolution (in seconds)
time0 = 0.0  # initial trajectory time
duration = 12 * 3600.  # duration of trajectory
alt = 40.0  # height for the trajectory

# Robot Params
vel = 0.5  # robot velocity (in meters/second)
samp_dist = 0.5  # distance between samples (in meters)
com_window = 120  # communication window (in seconds)
experiment_name = "crossflow_opt"

##################################################
# Create Environment and Model
##################################################
print("Generating environment and model classes.")

# Create Environment
env = CrossflowMTT(extent=extent, plume_loc=(0, 0), s=s,
                   tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0, area=a0, entrainment=E)

# Create Model
mtt = Crossflow(extent=extent, plume_loc=(0, 0), s=s, tprof=tprof, sprof=sprof,
                rhoprof=rhoprof, vex=v0_param, area=a0_param, density=rho0,
                curfunc=curfunc, headfunc=headfunc,
                salt=s0, temp=t0, E=(alph_param, bet_param))


# Draw a vertical slice of the plume envelope
mtt.solve(t=0.0)
le, cl, re = env.envelope(t=0.0)
plt.plot(*cl, label="Centerline")
plt.plot(*le, label="Left Extent")
plt.plot(*re, label="Right Extent")
plt.title("Plume Envelope")
plt.xlabel("X in crossflow direction (meters)")
plt.ylabel("Z (meters)")
plt.legend()

##################################################
# Create trajectory optimization object
##################################################
print("Generating trajectory optimizer...")

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)

fig, axs = plt.subplots(1, 1)

# Trajectory
planners = []
budget = time_resolution * vel  # distance budget per leg
for t, start_time in enumerate(np.arange(0, duration, step=time_resolution)):
    print(f"Initializing plan {t}")
    # Create the base trajectory generator object with a temporary start_point
    # and t0 value. We will reset these later in the chaining process.
    traj_generator = LawnSpiralWithStartGeneratorFlexible(
        t0=start_time, vel=vel, alt=alt, start_point=(0., 0.),
        traj_type=traj_type, res=traj_res)

    # Draw a vertical slice of the plume envelope
    mtt.solve(t=start_time)
    le, cl, re = mtt.odesys.envelope(t=start_time)
    axs.plot(*cl, label=f"t={start_time} Centerline")

    # Get predicted environment maxima
    # xm, ym, zm = mtt.get_maxima(start_time, z=[alt], xres=100, yres=100)
    # print(f"Maxima at time {start_time} is ({xm}, {ym})")

    # Create planner
    planners.append(TrajectoryOpt(
        mtt,
        traj_generator,
        reward,
        x0=(75., 75., 0., 0., 0.),  # (lh, lw, rot, origin_x, origin_y)
        param_bounds=[(20., 100), (20., 100.), (-360., 360.), extent.xrange, extent.yrange],
        param_names={"lh": 0, "lw": 1, "rot": 2, "origin_x": 3, "origin_y": 4},
        budget=budget,
        limits=[extent.xmin, extent.xmax, extent.ymin, extent.ymax],
        max_iters=1,
        tol=1e-2
    ))

planner = TrajectoryChain(planners=planners)

##################################################
# Get trajectory to execute
##################################################
print("Getting trajectory...")
tic()
plan_opt = planner.get_plan()
toc()

plt.plot(*plan_opt.path.xy, c='k')
plt.title("Trajectory")

##################################################
# Execute Trajectory in Real Environment to get Obs
##################################################
print("Executing trajectory in real environment...")

# Create the robot
rob = OfflineRobot(mtt, plan_opt, env, vel, com_window)

# Create the simulator
simulator = Simulator(rob, env, ref_global=True)

# Run the simulator
times = np.arange(0, duration + 1, 100)
tic()
simulator.simulate(times, experiment_name=experiment_name)
toc()

# Plot outcomes
simulator.plot_comms()
simulator.plot_all()
# simulator.plot_world()

mission_hash = get_mission_hash("mttcrossflow")
save_mission(mission_hash, rob, mtt, env, simulator)
print(f"Saving mission file to {mission_hash}")

plt.show()



