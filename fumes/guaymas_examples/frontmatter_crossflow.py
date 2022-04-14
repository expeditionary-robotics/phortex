"""Demonstrates the MTT model class prediction framework."""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from distfit import distfit
from fumes.model.mtt import MTT, Crossflow, Multimodel
from fumes.model.parameter import Parameter
from fumes.environment import CrossflowMTT
from fumes.environment.profile import Profile
from fumes.environment.current import CurrMag, CurrHead
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.environment.extent import Extent
from fumes.reward import SampleValues
from fumes.utils import tic, toc
from fumes.trajectory import Lawnmower, Spiral
from fumes.planner import TrajectoryOpt, TrajectoryChain, \
    LawnSpiralGeneratorFlexible, LawnSpiralWithStartGeneratorFlexible

from fumes.robot import OfflineRobot
from fumes.simulator import Simulator
from fumes.utils import save_mission, get_mission_hash, load_mission

from fumes.reward import SampleValues

# "Global" Parameters
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
alph = 0.12
bet = 0.20

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

alph_inf = distfit(distr='uniform')
alph_inf.fit_transform(np.random.uniform(0.1, 0.2, 2000))
alph_prop = sp.stats.norm(loc=0, scale=0.01)
alph_param = Parameter(alph_inf, alph_prop)

bet_inf = distfit(distr='uniform')
bet_inf.fit_transform(np.random.uniform(0.05, 0.25, 2000))
bet_prop = sp.stats.norm(loc=0, scale=0.05)
bet_param = Parameter(bet_inf, bet_prop)

# Current params
t = np.linspace(0, 12 * 3600, 50)
curmag = CurrMag(t, curfunc(None, t), training_iter=100, learning_rate=0.1)
curhead = CurrHead(t, headfunc(t) + np.random.normal(0, 0.01, t.shape),
                   training_iter=100, learning_rate=0.1)

# Simulation Params
xdim = 100  # target simulation xdim
ydim = 100  # target simulation ydim
zdim = 50
iter = 10  # number of samples to search over
thresh = 1e-5  # probability threshold for a detection

# Environment params
extent = Extent(xrange=(-500., 500.),
                xres=100,
                yrange=(-500., 500.),
                yres=100,
                zrange=(0., 200.),
                zres=100,
                global_origin=(27.407489, -111.389893, -1848.5))
s = np.linspace(0, 1000, 100)


# Parameter Optimization Params
iterations = 100  # number of samples to search over
burn = 20  # number of samples for burn-in

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

# Reward function
reward = SampleValues(
    sampling_params={"samp_dist": samp_dist},
    is_cost=True)

# Create Environment
env = CrossflowMTT(plume_loc=(0, 0, 0), extent=extent, s=s,
                   tprof=tprof.profile, sprof=sprof.profile,
                   rhoprof=rhoprof, vex=v0, area=a0,
                   density=rho0, curfunc=curmag.magnitude,
                   headfunc=curhead.heading,
                   salt=s0, temp=t0, entrainment=(alph, bet))
