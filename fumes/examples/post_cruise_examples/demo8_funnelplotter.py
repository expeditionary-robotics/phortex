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


# Set iteration parameters
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
v0_prior = 0.75
v0_update = 0.51
a0 = 0.1  # source area
a0_prior = 0.275
a0_update = 0.30
s0 = 34.608  # source salinity
t0 = 300  # source temperature
rho0 = eos_rho(t0, s0)  # source density
E = (0.12, 0.1)
E_prior = (0.15, 0.13)
E_update = (0.15, 0.11)

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

# Robot params
vel = 0.5  # robot velocity (in meters/second)
com_window = 120  # communication window (in seconds)
altitude = 70.0  # flight altitude (in meters)

# Create Environment
print("Creating true environment...")
env = CrossflowMTT(plume_loc=(0, 0, 0), extent=extent, s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0, area=a0, entrainment=E)

# Create Prior Model Snapshot
print("Creating prior environment...")
mtt = CrossflowMTT(plume_loc=(0, 0, 0), extent=extent, s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0_prior, area=a0_prior, entrainment=E_prior)

# Create updated Model Snapshot
print("Creating posterior environment...")
mttu = CrossflowMTT(plume_loc=(0, 0, 0), extent=extent, s=s,
                    tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                    density=rho0, salt=s0, temp=t0,
                    curfunc=curfunc, headfunc=headfunc,
                    vex=v0_update, area=a0_update, entrainment=E_update)

# plot the 3D funnels at t=0, just as a representation
layout = go.Layout(scene=dict(xaxis=dict(range=[-90, 20],),
                              yaxis=dict(range=[-10, 510],),
                              zaxis=dict(range=[0, 230],),
                              aspectmode='data'),)
env_3d_snapshot = env.get_pointcloud(t=0.)
env_fig = go.Scatter3d(x=env_3d_snapshot[:, 0],
                       y=env_3d_snapshot[:, 1],
                       z=env_3d_snapshot[:, 2],
                       mode="markers",
                       marker=dict(size=2, opacity=0.1, color='black'),
                       name=f"Environment Plume at t=0.")
mod_3d_snapshot = mtt.get_pointcloud(t=0.)
mod_fig = go.Scatter3d(x=mod_3d_snapshot[:, 0],
                       y=mod_3d_snapshot[:, 1],
                       z=mod_3d_snapshot[:, 2],
                       mode="markers",
                       marker=dict(size=2, opacity=0.1, color='blue'),
                       name=f"Model Plume at t=0.")
modu_3d_snapshot = mttu.get_pointcloud(t=0.)
new_mod_fig = go.Scatter3d(x=modu_3d_snapshot[:, 0],
                           y=modu_3d_snapshot[:, 1],
                           z=modu_3d_snapshot[:, 2],
                           mode="markers",
                           marker=dict(size=2, opacity=0.1, color='green'),
                           name=f"Updated Model Plume at t=0.")
fig = go.Figure(data=[env_fig, mod_fig, new_mod_fig], layout=layout)
fig.show()

x = np.linspace(-90, 20, 100)
y = np.linspace(-10, 510, 100)
X, Y = np.meshgrid(x, y)
plane = go.Surface(x=X, y=Y, z=70*np.ones_like(X))
fig = go.Figure(data=[env_fig, mod_fig, new_mod_fig, plane], layout=layout)
fig.show()
