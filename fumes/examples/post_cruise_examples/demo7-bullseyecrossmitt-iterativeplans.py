"""Demo script for model updates and trajectory optimization on HPC.

Models a crossflow world, with temporally varying crossflow.
"""

from curses import meta
import os
import numpy as np
import scipy as sp
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt

from fumes.environment.mtt import CrossflowMTT
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile
from fumes.environment.current import CurrMag, CurrHead
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, headfunc_fast

from fumes.model.mtt import Crossflow
from fumes.model.parameter import ParameterKDE

from fumes.reward import SampleValues

from fumes.robot import OfflineRobot
from fumes.simulator import Simulator

from fumes.planner import TrajectoryOpt, TrajectoryChain, LawnSpiralWithStartGeneratorFlexible
from fumes.utils.save_mission import save_experiment_json, save_experiment_visualsnapshot


# Set meta/saving parameters
code_test = False
experiment_name = f"local_bullseyemtt_iterativeplans_seed{np.random.randint(low=0, high=1000)}"
print("Experiment Name: ", experiment_name)

# Set iteration parameters
if code_test:
    sample_iter = 100  # number of samples to search over
    burn = 1  # number of burn-in samples
    plan_iter = 5  # planning iterations
    outer_iter = 2  # number of traj and model update loops
    samp_dist = 30.0  # distance between samples (in meters)
    time_resolution = 3600  # time resolution (in seconds)
    duration = 1 * 3600  # total mission time (in seconds)

else:
    sample_iter = 200  # number of samples to search over
    burn = 50  # number of burn-in samples
    plan_iter = 50  # planning iterations
    outer_iter = 5  # number of traj and model update loops
    samp_dist = 0.5  # distance between samples (in meters)
    time_resolution = 3600 * 2  # time resolution (in seconds)
    duration = 6 * 60 * 60  # total mission time (in seconds)

# "Global" Model Parameters
s = np.linspace(0, 500, 100)  # distance to integrate over
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
E = (0.12, 0.1)

# Inferred Source Params
v0_inf = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(
    np.random.uniform(0.05, 1.5, 5000)[:, np.newaxis])
v0_prop = sp.stats.norm(loc=0, scale=0.1)
v0_param = ParameterKDE(v0_inf, v0_prop, limits=(0.01, 3.0))

a0_inf = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(
    np.random.uniform(0.05, 0.5, 5000)[:, np.newaxis])
a0_prop = sp.stats.norm(loc=0, scale=0.1)
a0_param = ParameterKDE(a0_inf, a0_prop, limits=(0.01, 1.0))

alph_inf = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(
    np.random.uniform(0.1, 0.2, 5000)[:, np.newaxis])
alph_prop = sp.stats.norm(loc=0, scale=0.05)
alph_param = ParameterKDE(alph_inf, alph_prop, limits=(0.01, 0.3))

bet_inf = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(
    np.random.uniform(0.01, 0.25, 5000)[:, np.newaxis])
bet_prop = sp.stats.norm(loc=0, scale=0.05)
bet_param = ParameterKDE(bet_inf, bet_prop, limits=(0.01, 0.5))

# Current params
training_t = np.linspace(0, duration + 1, 100)
def curfunc(x, t): return np.ones_like(t) * 0.5  # set constant magnitude

directory = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/{experiment_name}")
if not os.path.exists(directory):
    os.makedirs(directory)
plt.plot(training_t / 3600. % 24., headfunc_fast(training_t) * 180. / np.pi + np.random.normal(0, 0.01, training_t.shape))
plt.xlabel('Time')
plt.ylabel('Current Heading')
plt.savefig(os.path.join(directory, f"current_function.png"))
plt.close()

curmag = CurrMag(training_t / 3600. % 24., curfunc(None, training_t) + np.random.normal(0, 0.01, training_t.shape),
                 training_iter=500, learning_rate=0.5)
curhead = CurrHead(training_t / 3600. % 24., headfunc_fast(training_t) * 180. / np.pi + np.random.normal(0, 0.01, training_t.shape),
                   training_iter=500, learning_rate=0.5)

# Model Simulation Params
extent = Extent(xrange=(-500., 500.),
                xres=200,
                yrange=(-500., 500.),
                yres=200,
                zrange=(0, 200),
                zres=50,
                global_origin=(0., 0., 0.))
thresh = 1e-5  # probability threshold for a detection
simulate_with_noise = True
simulator_noise = 0.1

# Trajectory params
traj_type = "lawnmower"  # type of fixed trajectory
resolution = 20  # lawnmower resolution (in meters)

# Robot params
vel = 1.0  # robot velocity (in meters/second)
com_window = 120  # communication window (in seconds)
altitude = 80.0  # flight altitude (in meters)

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)

# Create Environment
print("Creating true environment...")
env = CrossflowMTT(plume_loc=(0, 0, 0), extent=extent, s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc_fast,
                   vex=v0, area=a0, entrainment=E)

# Create Model
print("Creating estimated environment model...")
mtt = Crossflow(plume_loc=(0, 0, 0), extent=extent, s=s,
                tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                vex=v0_param, area=a0_param, density=rho0, salt=s0, temp=t0,
                curfunc=curmag, headfunc=curhead,
                E=(alph_param, bet_param))

meta_loop_data = []
for i in range(outer_iter):
    print("Starting to optimize...")
    # Append meta-iteration to the experiment name
    exp_name = f"{experiment_name}_metaloop{i}"
    mtt.experiment_name = exp_name

    # Build trajectory optimizer
    planners = []
    budget = time_resolution * vel  # distance budget per leg
    for start_time in np.arange(0, duration, step=time_resolution):
        # Create the base trajectory generator object with a temporary start_point
        # We will reset this later in the chaining process.
        print(f"Initializing trajectory generator at time {start_time}...")
        traj_generator = LawnSpiralWithStartGeneratorFlexible(
            t0=start_time, vel=vel, alt=altitude, start_point=(0., 0.),
            traj_type=traj_type, res=resolution)

        # Get predicted environment maxima
        xm, ym, zm = mtt.get_maxima(start_time, z=[altitude])

        # Create planner
        planners.append(TrajectoryOpt(
            mtt,
            traj_generator,
            reward,
            x0=(200., 200., 0., xm, ym),  # (lh, lw, rot, origin_x, origin_y)
            param_bounds=[(20., 500), (20., 500.), (-360., 360.), (-500., 500.), (-500., 500.)],
            param_names={"lh": 0, "lw": 1, "rot": 2, "origin_x": 3, "origin_y": 4},
            budget=budget,
            limits=[-500., 500., -500., 500.],
            max_iters=plan_iter,
            experiment_name=exp_name
        ))
        print("Done.")

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
    simulator.simulate(times,
                       experiment_name=f"{experiment_name}_iteration{i}",
                       with_noise=simulate_with_noise,
                       noise_portion=simulator_noise)

    # Update model
    print("Updating model!")
    obs = [[float(o > thresh)] for o in simulator.obs]
    obs = np.asarray(obs).flatten()
    print("Total samples: ", len(obs))
    print("Total obs: ", np.nansum(obs))
    obs_t = np.unique(np.round(times / 3600.))  # get snapshots by hour
    obs_c = []
    obs_o = []
    for j, ot in enumerate(obs_t):
        idt = np.round(times / 3600.) == ot
        obs_c.append((simulator.coords[idt, 0], simulator.coords[idt, 1], simulator.coords[idt, 2]))
        obs_o.append(obs[idt])
    newAlph, newBet, newVelocity, newArea = mtt.update(obs_t * 3600.,
                                                       np.asarray(obs_c),
                                                       np.asarray(obs_o),
                                                       num_samps=sample_iter,
                                                       burnin=burn,
                                                       thresh=thresh)
    print(f"E_alpha:{newAlph}, E_beta:{newBet}, V:{newVelocity}, A:{newArea}")

    # Log information
    experiment_dict = {"experiment_iteration": i,
                       "total_experiment_iterations": outer_iter,
                       "total_samples": len(obs),
                       "in_plume_thresh": thresh,
                       "total_in_plume_samples": np.nansum(obs),
                       "portion_in_plume_samples": float(np.nansum(obs) / len(obs))}

    meta_loop_data.append(experiment_dict)

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

    save_experiment_visualsnapshot(experiment_name,
                                   iter_num=i,
                                   rob=rob,
                                   model=mtt,
                                   env=env,
                                   traj_opt=planners,
                                   trajectory=plan_opt,
                                   reward=reward,
                                   simulation=simulator,
                                   experiment_dict=experiment_dict)

    directory = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/{experiment_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.close()
    fig, ax = plt.subplots(2, 1)
    ax[0].scatter(range(len(meta_loop_data)), [d["total_in_plume_samples"] for d in meta_loop_data])
    ax[1].scatter(range(len(meta_loop_data)), [d["portion_in_plume_samples"] for d in meta_loop_data])
    plt.savefig(os.path.join(directory, f"meta_loop_reward.png"))
    plt.close()
