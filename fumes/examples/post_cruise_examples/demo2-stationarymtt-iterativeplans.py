"""Demo script for model updates and trajectory optimization on HPC."""

# Note: draws heavily from
# pre_cruise_examples > stationary_mtt_sample_and_udpate.py

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from distfit import distfit

from fumes.environment.mtt import StationaryMTT
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S

from fumes.model.mtt import MTT
from fumes.model.parameter import Parameter

from fumes.reward import SampleValues

from fumes.robot import OfflineRobot
from fumes.simulator import Simulator

from fumes.planner import TrajectoryOpt, TrajectoryChain, LawnSpiralWithStartGeneratorFlexible
from fumes.utils.save_mission import save_experiment_json


# Set meta/saving parameters
code_test = True
experiment_name = "stationarymtt_iterativeplans"

# Set iteration parameters
if code_test is True:
    sample_iter = 5  # number of samples to search over
    burn = 1  # number of burn-in samples
    plan_iter = 1  # planning iterations
    outer_iter = 2  # number of traj and model update loops
    samp_dist = 10.0  # distance between samples (in meters)
    time_resolution = 100  # time resolution (in seconds)
    duration = 2 * 100  # total mission time (in seconds)

else:
    sample_iter = 100  # number of samples to search over
    burn = 50  # number of burn-in samples
    plan_iter = 100  # planning iterations
    outer_iter = 5  # number of traj and model update loops
    samp_dist = 0.5  # distance between samples (in meters)
    time_resolution = 3600 * 4  # time resolution (in seconds)
    duration = 8 * 60 * 60  # total mission time (in seconds)

# "Global" Model Parameters
z = np.linspace(0, 200, 100)  # height to integrate over
tprof = Profile(z, pacific_sp_T(z))  # function that describes background temp
sprof = Profile(z, pacific_sp_S(z))  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T

# True Source Params
v0 = 0.4  # source exit velocity
a0 = 0.1  # source area
s0 = 34.608  # source salinity
t0 = 300  # source temperature
rho0 = eos_rho(t0, s0)  # source density
E = 0.255

# Inferred Source Params
v0_inf = distfit(distr='uniform')
v0_inf.fit_transform(np.random.uniform(0.05, 1.5, 2000))
v0_prop = sp.stats.norm(loc=0, scale=0.05)
v0_param = Parameter(v0_inf, v0_prop)

a0_inf = distfit(distr='uniform')
a0_inf.fit_transform(np.random.uniform(0.05, 0.5, 2000))
a0_prop = sp.stats.norm(loc=0, scale=0.01)
a0_param = Parameter(a0_inf, a0_prop)

E_inf = distfit(distr='uniform')
E_inf.fit_transform(np.random.uniform(0.1, 0.4, 2000))
E_prop = sp.stats.norm(loc=0, scale=0.01)
E_param = Parameter(E_inf, E_prop)

# Model Simulation Params
extent = Extent(xrange=(-500., 500.),
                xres=200,
                yrange=(-500., 500.),
                yres=200,
                zrange=(0, 200),
                zres=50,
                global_origin=(0., 0., 0.))
thresh = 1e-5  # probability threshold for a detection

# Trajectory params
traj_type = "lawnmower"  # type of fixed trajectory
resolution = 5  # lawnmower resolution (in meters)

# Robot params
vel = 0.5  # robot velocity (in meters/second)
com_window = 120  # communication window (in seconds)
altitude = 150.0  # flight altitude (in meters)

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)

# Create Environment
print("Creating true environment...")
env = StationaryMTT(plume_loc=(0, 0, 0), extent=extent, z=z,
                    tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                    density=rho0, salt=s0, temp=t0,
                    vex=v0, area=a0, entrainment=E)

# Create Model
print("Creating estimated environment model...")
mtt = MTT(plume_loc=(0, 0, 0), extent=extent, z=z, tprof=tprof, sprof=sprof, rhoprof=rhoprof,
          vex=v0_param, area=a0_param, density=rho0, salt=s0, temp=t0,
          E=E_param)

for i in range(outer_iter):
    print("Starting to optimize...")
    # Build trajectory optimizer
    planners = []
    budget = time_resolution * vel  # distance budget per leg
    for start_time in np.arange(0, duration, step=time_resolution):
        # Create the base trajectory generator object with a temporary start_point
        # We will reset this later in the chaining process.
        print("Trajectory generator generating...")
        traj_generator = LawnSpiralWithStartGeneratorFlexible(
            t0=start_time, vel=vel, alt=altitude, start_point=(0., 0.),
            traj_type=traj_type, res=resolution)

        # Get predicted environment maxima
        xm, ym, zm = mtt.get_maxima(start_time, z=[altitude])

        # Create planner
        print("Trajectory optimizer generating...")
        planners.append(TrajectoryOpt(
            mtt,
            traj_generator,
            reward,
            x0=(200., 200., 0., xm, ym),  # (lh, lw, rot, origin_x, origin_y)
            param_bounds=[(20., 500), (20., 500.), (-360., 360.), (-500., 500.), (-500., 500.)],
            param_names={"lh": 0, "lw": 1, "rot": 2, "origin_x": 3, "origin_y": 4},
            budget=budget,
            limits=[-500., 500., -500., 500.],
            max_iters=plan_iter
        ))

    print("Planners created!")
    planner = TrajectoryChain(planners=planners)
    print("Planners chained! Now getting plan...")
    plan_opt = planner.get_plan()
    print("Plan in place!")

    # Create the robot
    rob = OfflineRobot(mtt, plan_opt, env, vel, com_window)
    print("Created robot!")

    # Create the simulator
    simulator = Simulator(rob, env)
    print("Simulating...")

    # Run the simulator
    times = np.arange(0, duration + 1)
    simulator.simulate(times, experiment_name=f"{experiment_name}_iteration{i}")

    # Plot outcomes
    # print("Plotting simulations...")
    # simulator.plot_comms()
    # simulator.plot_all()
    # simulator.plot_world(frame_skip=3600)

    # Update model
    print("Updating model!")
    obs = [float(o > thresh) for o in simulator.obs]
    update_coords = [simulator.coords.T[0], simulator.coords.T[1], simulator.coords.T[2]]
    newEntrainment, newVelocity, newArea = mtt.update(0.0,  # since MTT is stationary, just post a single time
                                                      update_coords,
                                                      obs,
                                                      num_samps=sample_iter,
                                                      burnin=burn,
                                                      thresh=thresh)
    print(newEntrainment, newVelocity, newArea)

    # Log information
    experiment_dict = {"experiment_iteration": i,
                       "total_experiment_iterations": outer_iter,
                       "total_samples": len(obs),
                       "in_plume_thresh": thresh,
                       "total_in_plume_samples": np.nansum(obs),
                       "portion_in_plume_samples": float(np.nansum(obs) / len(obs))}
    
    save_experiment_json(experiment_name,
                         iter_num=i,
                         rob=rob,
                         model=mtt,
                         env=env,
                         traj_opt=planners[0],
                         trajectory=plan_opt,
                         reward=reward,
                         simulation=simulator,
                         experiment_dict=experiment_dict)

# Generate simple visualizations
