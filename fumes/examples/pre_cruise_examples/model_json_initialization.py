"""Demonstrates initializing a model from JSON."""

import numpy as np
import scipy as sp
import os
import matplotlib.pyplot as plt
from distfit import distfit
from fumes.model.mtt import MTT, Crossflow, Multimodel
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.environment.extent import Extent
from fumes.model.parameter import Parameter
from fumes.environment.current import CurrHead, CurrMag
from fumes.environment.profile import Profile
from fumes.model.saving_utils import initialize_model_from_json
from fumes.utils import data_home

# "Global" Parameters
z = np.linspace(0, 200, 100)  # height to integrate over
tprof = Profile(z, pacific_sp_T(z))  # function that describes background temp
sprof = Profile(z, pacific_sp_S(z))  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T
T = np.linspace(0, 12*3600, 1000)

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

extent = Extent(xrange=(0, 500),
                xres=50,
                yrange=(0, 500),
                yres=30,
                zrange=(0, 50),
                zres=10)

# Current params
curmag = CurrMag(T, curfunc(None, T), training_iter=100, learning_rate=0.1)
curhead = CurrHead(T, np.sin(T/3600), training_iter=100, learning_rate=0.1)


RUN_STATIONARY = True
RUN_CROSSFLOW = True
RUN_MULTIPLUME = True

####################
# Stationary Model
####################
if RUN_STATIONARY is True:
    print("Running Stationary Example...")
    # create model
    mtt = MTT(plume_loc=(0, 0, 0), extent=extent, z=z, tprof=tprof,
              sprof=sprof, rhoprof=rhoprof, vex=v0_param, area=a0_param,
              density=rho0, salt=s0, temp=t0, E=E_param)

    # save model
    print("Saving Stationary...")
    mtt.save_model_metadata()

    # now create new model
    print("Initializing Stationary...")
    init = initialize_model_from_json(os.path.join(data_home(), f"{mtt.NAME}.json"))
    mtt_from_file = MTT(**init)

    # compare models
    mtt.get_parameters()
    mtt_from_file.get_parameters()

####################
# Crossflow Model
####################
if RUN_CROSSFLOW is True:
    print("Running Crossflow Example...")
    # create model
    s = np.linspace(0, 1000, 100)
    mtt = Crossflow(plume_loc=(0, 0, 0), extent=extent, s=s,
                    tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                    vex=v0_param, area=a0_param, density=rho0,
                    curfunc=curmag, headfunc=curhead,
                    salt=s0, temp=t0, E=(alph_param, bet_param))

    # save model
    print("Saving Crossflow...")
    mtt.save_model_metadata()

    # now create new model
    print("Initializing Crossflow from file...")
    init = initialize_model_from_json(os.path.join(data_home(), f"{mtt.NAME}.json"))
    mtt_from_file = Crossflow(**init)

    # compare models
    mtt.get_parameters()
    mtt_from_file.get_parameters()

    le, cl, re = mtt.odesys.envelope(t=0.0)
    fle, fcl, fre = mtt_from_file.odesys.envelope(t=0.0)

    plt.plot(*cl, label="Centerline")
    plt.plot(*le, label="Left")
    plt.plot(*re, label="Right")
    plt.plot(*fcl, label="Centerline File")
    plt.plot(*fle, label="Left File")
    plt.plot(*fre, label="Right File")
    plt.xlabel("X coords")
    plt.ylabel("Z coords")
    plt.legend()
    plt.show()

    temp = mtt.odesys.tprof(z)
    salt = mtt.odesys.sprof(z)
    plt.plot(mtt.odesys.rhoprof(temp, salt))
    plt.plot(mtt_from_file.odesys.rhoprof(temp, salt))
    plt.show()

####################
# Multiplume Model
####################
if RUN_MULTIPLUME is True:
    print("Running Multiplume Example...")
    # create model
    s = np.linspace(0, 1000, 100)
    mtt1 = Crossflow(plume_loc=(0, 0, 0), extent=extent, s=s,
                     tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                     vex=v0_param, area=a0_param, density=rho0,
                     curfunc=curmag, headfunc=curhead,
                     salt=s0, temp=t0, E=(alph_param, bet_param))
    mtt2 = Crossflow(plume_loc=(100, 0, 0), extent=extent, s=s,
                     tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                     vex=v0_param, area=a0_param, density=rho0,
                     curfunc=curmag, headfunc=curhead,
                     salt=s0, temp=t0, E=(alph_param, bet_param))
    mtt = Multimodel([mtt1, mtt2])

    # save model
    mtt.save_model_metadata()
    multiplume_name = mtt.NAME

    # now create new model
    mods = []
    for m in mtt.models:
        mname = m.NAME
        init = initialize_model_from_json(os.path.join(
            data_home(), f"{mname}_multipart_{multiplume_name}.json"))
        mods.append(Crossflow(**init))

    mtt_from_file = Multimodel(mods)

    # compare models
    mtt.models[0].get_parameters()
    mtt_from_file.models[0].get_parameters()
