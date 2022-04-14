"""Demonstrates the MTT model class prediction framework."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from distfit import distfit
from fumes.model.mtt import MTT, Crossflow, Multimodel
from fumes.model.parameter import Parameter
from fumes.environment.current import CurrMag, CurrHead
from fumes.environment.extent import Extent
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc

# Simulation Params
xdim = 100  # target simulation xdim
ydim = 100  # target simulation ydim
zdim = 100
T = np.linspace(0, 12 * 3600, 10)  # time snapshots
iter = 100  # number of samples to search over
thresh = 1e-5  # probability threshold for a detection

RUN_STATIONARY = True
RUN_CROSSFLOW = True
RUN_MULTIPLUME = True

# "Global" Parameters
z = np.linspace(0, 200, 100)  # height to integrate over
tprof = pacific_sp_T  # function that describes background temp
sprof = pacific_sp_S  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T
loc = (0, 0, 0)  # plume location
extent = Extent(xrange=(0, xdim),
                xres=xdim,
                yrange=(0, ydim),
                yres=ydim,
                zrange=None,
                zres=None,
                global_origin=loc)

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
headfunc = lambda t: np.sin(t/3600.)
head = np.sin(T/3600)
cur = curfunc(None, T)
curmag = CurrMag(T, cur, training_iter=100, learning_rate=0.1)
curhead = CurrHead(T, head, training_iter=100, learning_rate=0.1)


####################
# Stationary Model
####################
if RUN_STATIONARY is True:
    print("Running Stationary Example...")
    mtt = MTT(plume_loc=loc, z=z, tprof=tprof, sprof=sprof, rhoprof=rhoprof,
              vex=v0_param, area=a0_param, density=rho0, salt=s0, temp=t0,
              E=E_param, extent=extent)

    mtt.save_model_metadata(overwrite=True)

####################
# Crossflow Model
####################
if RUN_CROSSFLOW is True:
    print("Running Crossflow Example...")
    s = np.linspace(0, 1000, 100)
    mtt = Crossflow(plume_loc=loc, s=s, tprof=tprof, sprof=sprof,
                    rhoprof=rhoprof, vex=v0_param, area=a0_param, density=rho0,
                    curfunc=curmag, headfunc=curhead,
                    salt=s0, temp=t0, E=(alph_param, bet_param), extent=extent)

    mtt.save_model_metadata(overwrite=True)

####################
# Crossflow Model
####################
if RUN_MULTIPLUME is True:
    print("Running Multiplume Example...")
    s = np.linspace(0, 1000, 100)
    mtt1 = Crossflow(plume_loc=loc, s=s, tprof=tprof, sprof=sprof,
                     rhoprof=rhoprof, vex=v0_param, area=a0_param,
                     density=rho0, curfunc=curmag, headfunc=curhead,
                     salt=s0, temp=t0, E=(alph_param, bet_param),
                     extent=extent)
    mtt2 = Crossflow(plume_loc=(75, 0, 0), s=s, tprof=tprof, sprof=sprof,
                     rhoprof=rhoprof, vex=v0_param, area=a0_param,
                     density=rho0, curfunc=curmag, headfunc=curhead,
                     salt=s0, temp=t0, E=(alph_param, bet_param),
                     extent=extent)
    mtt = Multimodel([mtt1, mtt2])

    mtt.save_model_metadata(overwrite=True)
