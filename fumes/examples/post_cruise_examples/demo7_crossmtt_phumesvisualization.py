"""Demo script for model updates and trajectory optimization on HPC.

Models a crossflow world, with temporally varying crossflow.
"""

import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from fumes.simulator.utils import scatter_obj
from fumes.trajectory.lawnmower import Lawnmower
from fumes.trajectory.trajectory import Trajectory
from sklearn.neighbors import KernelDensity

from fumes.environment.mtt import CrossflowMTT
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile
from fumes.environment.current import CurrMag, CurrHead
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, curfunc, headfunc

from fumes.model.mtt import Crossflow
from fumes.model.parameter import ParameterKDE

from fumes.reward import SampleValues

from fumes.robot import OfflineRobot
from fumes.simulator import Simulator

from fumes.utils.save_mission import save_experiment_json, save_experiment_visualsnapshot_atT

# E_alpha:0.14731761054041068, E_beta:0.10540189607026254, V:0.5144557054857927, A:0.3028688599354139
# E_alpha:0.15026115417029667, E_beta:0.13921694335379634, V:0.6672301198899803, A:0.21518419081437928

# Set meta/saving parameters
code_test = False
experiment_name = f"tests_local_phumes_seed{np.random.randint(low=0, high=1000)}"
print("Experiment Name: ", experiment_name)
directory = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/{experiment_name}")
model_directory = os.path.join(os.getenv("FUMES_OUTPUT"), f"modeling/{experiment_name}")
if not os.path.exists(directory):
    os.makedirs(directory)
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Set iteration parameters
if code_test:
    sample_iter = 50  # number of samples to search over
    burn = 1  # number of burn-in samples
    samp_dist = 0.5  # distance between samples (in meters)
    time_resolution = 3600  # time resolution (in seconds)
    duration = 1 * 3600  # total mission time (in seconds)
    num_snaps = 2

else:
    sample_iter = 200  # number of samples to search over
    burn = 50  # number of burn-in samples
    samp_dist = 0.5  # distance between samples (in meters)
    time_resolution = 3600  # time resolution (in seconds)
    duration = 12 * 3600  # total mission time (in seconds)
    num_snaps = 12

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

#####
# Plot Initial Source Params
#####
plt.plot(np.linspace(0, 1, 100), alph_param.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
plt.vlines(E[0], 0, 10, colors="red", linestyles="--")
plt.xlabel("Alpha Sample Values")
plt.ylabel("PDF")
plt.title("Alpha Samples")
plt.savefig(os.path.join(model_directory, "alpha_distribution_init.svg"))
plt.close()

plt.plot(np.linspace(0, 1, 100), bet_param.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
plt.vlines(E[1], 0, 10, colors="red", linestyles="--")
plt.xlabel("Beta Sample Values")
plt.ylabel("PDF")
plt.title("Beta Samples")
plt.savefig(os.path.join(model_directory, "beta_distribution_init.svg"))
plt.close()

plt.plot(np.linspace(0, 1, 100), v0_param.predict(np.linspace(0, 2, 100)), linewidth=3, alpha=0.5)
plt.vlines(v0, 0, 10, colors="red", linestyles="--")
plt.xlabel("Velocity Sample Values")
plt.ylabel("PDF")
plt.title("Velocity Samples")
plt.savefig(os.path.join(model_directory, "velocity_distribution_init.svg"))
plt.close()

plt.plot(np.linspace(0, 1, 100), a0_param.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
plt.vlines(a0, 0, 10, colors="red", linestyles="--")
plt.xlabel("Area Sample Values")
plt.ylabel("PDF")
plt.title("Area Samples")
plt.savefig(os.path.join(model_directory, "area_distribution.svg"))
plt.close()

# Current params
training_t = np.linspace(0, duration + 1, 100)
curmag = CurrMag(training_t / 3600. % 24., curfunc(None, training_t) + np.random.normal(0, 0.01, training_t.shape),
                 training_iter=500, learning_rate=0.5)
curhead = CurrHead(training_t / 3600. % 24., headfunc(training_t) * 180. / np.pi + np.random.normal(0, 0.01, training_t.shape),
                   training_iter=500, learning_rate=0.5)

# Model Simulation Params
extent = Extent(xrange=(0., 500.),
                xres=100,
                yrange=(0., 500.),
                yres=100,
                zrange=(0, 200),
                zres=50,
                global_origin=(0., 0., 0.))
thresh = 1e-5  # probability threshold for a detection
simulate_with_noise = False
simulator_noise = 0.1
snap_times = np.linspace(0, duration + 1, num_snaps)

# Trajectory params
traj_type = "lawnmower"  # type of fixed trajectory
resolution = 15.  # lawnmower resolution (in meters)

# Robot params
vel = 0.5  # robot velocity (in meters/second)
com_window = 120  # communication window (in seconds)
altitude = 70.0  # flight altitude (in meters)

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)

# Create Environment
print("Creating true environment...")
env = CrossflowMTT(plume_loc=(0, 0, 0), extent=extent, s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0, area=a0, entrainment=E)

# Create Model
print("Creating estimated environment model...")
mtt = Crossflow(plume_loc=(0, 0, 0), extent=extent, s=s,
                tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                vex=v0_param, area=a0_param, density=rho0, salt=s0, temp=t0,
                curfunc=curmag, headfunc=curhead,
                E=(alph_param, bet_param))

################
# Model and Environment Initialization Plots and Saving
################
for st in snap_times:
    # plot underlying environment
    env_snapshot = env.get_snapshot(t=st, z=[altitude], from_cache=False)
    plt.imshow(env_snapshot[0], origin="lower", extent=(env.extent.xrange[0],
                                                        env.extent.xrange[1], env.extent.yrange[0], env.extent.yrange[1]))
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title("Environment Snapshot")
    plt.savefig(os.path.join(directory, f"env_snapshot_t{round(st)}_init.png"))
    plt.close()

    # plot learned model
    mod_snapshot = mtt.get_snapshot(t=st, z=[altitude], from_cache=False)
    plt.imshow(mod_snapshot[0], origin="lower", extent=(env.extent.xrange[0],
                                                        env.extent.xrange[1], env.extent.yrange[0], env.extent.yrange[1]))
    plt.xlabel('X-coordinate')
    plt.ylabel('Y-coordinate')
    plt.title("Model Snapshot")
    plt.savefig(os.path.join(directory, f"model_snapshot_t{round(st)}_init.png"))
    plt.close()

# plot the 3D funnels at t=0, just as a representation
layout = go.Layout(scene=dict(aspectmode='data'))
env_3d_snapshot = env.get_pointcloud(t=0.)
env_fig = go.Scatter3d(x=env_3d_snapshot[:, 0],
                       y=env_3d_snapshot[:, 1],
                       z=env_3d_snapshot[:, 2],
                       mode="markers",
                       marker=dict(size=0.5, opacity=0.7, color='green'),
                       name=f"Environment Plume at t=0.")
mod_3d_snapshot = mtt.odesys.get_pointcloud(t=0.)
mod_fig = go.Scatter3d(x=mod_3d_snapshot[:, 0],
                       y=mod_3d_snapshot[:, 1],
                       z=mod_3d_snapshot[:, 2],
                       mode="markers",
                       marker=dict(size=0.5, opacity=0.7, color='blue'),
                       name=f"Model Plume at t=0.")

print("Starting to optimize...")
# Append meta-iteration to the experiment name
exp_name = f"{experiment_name}_metaloop1"
mtt.experiment_name = exp_name

plan_opt = Lawnmower(t0=0,
                     vel=vel,
                     lh=500,
                     lw=500,
                     resolution=15,
                     altitude=altitude,
                     origin=(0, 500),
                     orientation=-90.,
                     noise=None)
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
                   experiment_name=f"{experiment_name}_iteration1",
                   with_noise=simulate_with_noise,
                   noise_portion=simulator_noise)

# Update model
print("Updating model!")
obs = [[float(o > thresh)] for o in simulator.obs]
obs = np.asarray(obs).flatten()
print("Total samples: ", len(obs))
print("Total obs: ", np.nansum(obs))
obs_t = np.unique(np.round(times / 3600.))  # get snapshots by hour
print(obs_t)
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

mod_3d_snapshot = mtt.odesys.get_pointcloud(t=0.)
new_mod_fig = go.Scatter3d(x=mod_3d_snapshot[:, 0],
                           y=mod_3d_snapshot[:, 1],
                           z=mod_3d_snapshot[:, 2],
                           mode="markers",
                           marker=dict(size=0.5, opacity=0.7, color='lightpink'),
                           name=f"Updated Model Plume at t=0.")
fig = go.Figure(data=[env_fig, mod_fig, new_mod_fig], layout=layout)

# Log information
experiment_dict = {"experiment_iteration": 1,
                   "total_experiment_iterations": 1,
                   "total_samples": len(obs),
                   "in_plume_thresh": thresh,
                   "total_in_plume_samples": np.nansum(obs),
                   "portion_in_plume_samples": float(np.nansum(obs) / len(obs))}

save_experiment_json(experiment_name,
                     iter_num=1,
                     rob=rob,
                     model=mtt,
                     env=env,
                     traj_opt=None,
                     trajectory=plan_opt,
                     reward=reward,
                     simulation=simulator,
                     experiment_dict=experiment_dict)

save_experiment_visualsnapshot_atT(experiment_name,
                                   iter_num=1,
                                   rob=rob,
                                   model=mtt,
                                   env=env,
                                   traj_opt=None,
                                   trajectory=plan_opt,
                                   reward=reward,
                                   simulation=simulator,
                                   experiment_dict=experiment_dict,
                                   T=np.linspace(0, duration + 1, 12))
fig.show()