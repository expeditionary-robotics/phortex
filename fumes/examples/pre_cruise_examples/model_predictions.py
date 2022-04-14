"""Demonstrates the MTT model class prediction framework."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from distfit import distfit
from fumes.model.mtt import MTT, Crossflow, Multimodel
from fumes.model.parameter import Parameter
from fumes.environment.profile import Profile
from fumes.environment.current import CurrMag, CurrHead
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.environment.extent import Extent

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
t = np.linspace(0, 12*3600, 50)
curmag = CurrMag(t, curfunc(None, t), training_iter=100, learning_rate=0.1)
curhead = CurrHead(t, headfunc(t)+np.random.normal(0, 0.01, t.shape), training_iter=100, learning_rate=0.1)

# Simulation Params
xdim = 100  # target simulation xdim
ydim = 100  # target simulation ydim
zdim = 50
T = np.linspace(0, 12 * 3600, 10)  # time snapshots
iter = 10  # number of samples to search over
thresh = 1e-5  # probability threshold for a detection

RUN_STATIONARY = False
RUN_CROSSFLOW = True
RUN_MULTIPLUME = True
MAKE_CACHES = False

####################
# Stationary Model
####################
if RUN_STATIONARY is True:
    print("Running Stationary Example...")
    extent = Extent(xrange=(-xdim / 2, xdim / 2),
                    xres=xdim,
                    yrange=(-ydim / 2, ydim / 2),
                    yres=ydim,
                    zrange=(0, 50),
                    zres=zdim)
    mtt = MTT(plume_loc=(0, 0, 0), extent=extent, z=z, tprof=tprof,
              sprof=sprof, rhoprof=rhoprof, vex=v0_param, area=a0_param,
              density=rho0, salt=s0, temp=t0, E=E_param)

    y = np.linspace(-100, 100, 1000)
    x = np.zeros_like(y)
    height = np.ones_like(x) * 35.
    pq = mtt.get_value(t=0.0, loc=(x, y, height))
    plt.plot(y, pq, c='r')
    for i in range(iter):
        mq, vq = mtt.get_prediction(t=0.0, loc=(x, y, height), num_samples=1)
        plt.plot(y, mq, c='k', alpha=0.1)
    plt.xlabel('Y-coordinate')
    plt.ylabel('Prob Value')
    plt.title('Environment Slice at X=0')
    plt.show()

    # Now get snapshot
    mean_snap = mtt.get_snapshot(t=0, z=[35.], from_cache=False)

    mean_pred, var_pred = mtt.get_snapshot_prediction(t=0,
                                                      z=[35.],
                                                      from_cache=False,
                                                      num_samples=iter)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(mean_snap[0], origin="lower", extent=(-xdim / 2, xdim / 2,
                                                       -ydim / 2, ydim / 2))
    ax[0].set_xlabel('X-coordinate')
    ax[0].set_ylabel('Y-coordinate')
    ax[0].set_title("Snapshot")
    ax[1].imshow(mean_pred[0], origin="lower", extent=(-xdim / 2, xdim / 2,
                                                       -ydim / 2, ydim / 2))
    ax[1].set_xlabel('X-coordinate')
    ax[1].set_ylabel('Y-coordinate')
    ax[1].set_title("Pred Mean")
    ax[2].imshow(var_pred[0], origin="lower", extent=(-xdim / 2, xdim / 2,
                                                      -ydim / 2, ydim / 2))
    ax[2].set_xlabel('X-coordinate')
    ax[2].set_ylabel('Y-coordinate')
    ax[2].set_title("Pred Var")
    plt.show()

    if MAKE_CACHES is True:
        print("Writing cache...")
        mtt.write_prediction_cache(tvec=[0],
                                   xrange=(-xdim / 2, xdim / 2),
                                   yrange=(-ydim / 2, ydim / 2),
                                   zrange=(0, zdim),
                                   xres=xdim,
                                   yres=ydim,
                                   zres=zdim,
                                   overwrite=True,
                                   num_samples=iter)

        print("Reading from cache...")
        mean_pred, var_pred = mtt.get_snapshot_prediction(t=0,
                                                          z=[0, 10, 20, 30, 40, 50],
                                                          from_cache=True,
                                                          num_samples=iter)

        fig, ax = plt.subplots(6, 2, sharex=True, sharey=True)
        heights = [0, 10, 20, 30, 40, 50]
        for i in range(6):
            ax[i, 0].imshow(mean_pred[i], origin="lower",
                            extent=(-xdim / 2, xdim / 2, -ydim / 2, ydim / 2))
            ax[i, 0].set_xlabel('X-coordinate')
            ax[i, 0].set_ylabel('Y-coordinate')
            ax[i, 0].set_title("Pred Mean, z=" + str(heights[i]))
            ax[i, 1].imshow(var_pred[i], origin="lower",
                            extent=(-xdim / 2, xdim / 2, -ydim / 2, ydim / 2))
            ax[i, 1].set_xlabel('X-coordinate')
            ax[i, 1].set_ylabel('Y-coordinate')
            ax[i, 1].set_title("Pred Var")
        plt.show()

    # Get a plume intersection of points
    x = np.linspace(-xdim / 2, xdim / 2, xdim)
    y = np.zeros_like(x)
    mean = np.zeros((zdim, xdim))
    var = np.zeros((zdim, xdim))
    for i, zh in enumerate(np.linspace(0, 50, zdim)):
        print(zh)
        height = np.ones_like(x) * zh
        mq, vq = mtt.get_prediction(t=0.0,
                                    loc=(x, y, height),
                                    num_samples=iter,
                                    from_cache=False)
        mean[i, :] = mq
        var[i, :] = vq
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(mean, origin="lower", extent=(-xdim / 2, xdim / 2, 0, zdim))
    ax[0].set_xlabel('X-coordinate')
    ax[0].set_ylabel('Z-coordinate')
    ax[0].set_title('Mean Pred')
    ax[1].imshow(var/mean**2, origin="lower", extent=(-xdim / 2, xdim / 2, 0, zdim))
    ax[1].set_xlabel('X-coordinate')
    ax[1].set_ylabel('Z-coordinate')
    ax[1].set_title('Var Pred')
    plt.show()

####################
# Crossflow Model
####################
if RUN_CROSSFLOW is True:
    print("Running Crossflow Example...")
    s = np.linspace(0, 1000, 100)
    extent = Extent(xrange=(0, xdim),
                    xres=xdim,
                    yrange=(0, ydim),
                    yres=ydim,
                    zrange=(0, 50),
                    zres=zdim)
    mtt = Crossflow(plume_loc=(0, 0, 0), extent=extent,
                    s=s, tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                    vex=v0_param, area=a0_param, density=rho0,
                    curfunc=curmag, headfunc=curhead,
                    salt=s0, temp=t0, E=(alph_param, bet_param))

    # Get a plume intersection of points
    y = np.linspace(-100, 100, 1000)
    x = np.zeros_like(y)
    height = np.ones_like(x) * 35.
    pq = mtt.get_value(t=0.0, loc=(x, y, height))
    plt.plot(y, pq, c='r')
    for i in range(iter):
        mq, vq = mtt.get_prediction(t=0.0, loc=(x, y, height), num_samples=1)
        plt.plot(y, mq, c='k', alpha=0.1)
    plt.xlabel('Y-coordinate')
    plt.ylabel('Prob Value')
    plt.title('Environment Slice at X=0')
    plt.show()

    # Now get snapshot
    mean_snap = mtt.get_snapshot(t=0, z=[35.], from_cache=False)

    mean_pred, var_pred = mtt.get_snapshot_prediction(t=0,
                                                      z=[35.],
                                                      from_cache=False,
                                                      num_samples=iter)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(mean_snap[0], origin="lower", extent=(0, xdim, 0, ydim))
    ax[0].set_xlabel('X-coordinate')
    ax[0].set_ylabel('Y-coordinate')
    ax[0].set_title("Snapshot")
    ax[1].imshow(mean_pred[0], origin="lower", extent=(0, xdim, 0, ydim))
    ax[1].set_xlabel('X-coordinate')
    ax[1].set_ylabel('Y-coordinate')
    ax[1].set_title("Pred Mean")
    ax[2].imshow(var_pred[0], origin="lower", extent=(0, xdim, 0, ydim))
    ax[2].set_xlabel('X-coordinate')
    ax[2].set_ylabel('Y-coordinate')
    ax[2].set_title("Pred Var")
    plt.show()

    if MAKE_CACHES is True:
        print("Writing cache...")
        mtt.write_prediction_cache(tvec=[0],
                                   xrange=(0, xdim),
                                   yrange=(0, ydim),
                                   zrange=(0, zdim),
                                   xres=xdim,
                                   yres=ydim,
                                   zres=zdim,
                                   overwrite=True,
                                   num_samples=iter)

        print("Reading from cache...")
        mean_pred, var_pred = mtt.get_snapshot_prediction(t=0,
                                                          z=[0, 10, 20, 30, 40, 50],
                                                          from_cache=True,
                                                          num_samples=iter)

        fig, ax = plt.subplots(6, 3, sharex=True, sharey=True)
        heights = [0, 10, 20, 30, 40, 50]
        for i in range(6):
            ax[i, 0].imshow(mean_pred[i], origin="lower", extent=(0, xdim, 0, ydim))
            ax[i, 0].set_xlabel('X-coordinate')
            ax[i, 0].set_ylabel('Y-coordinate')
            ax[i, 0].set_title("Pred Mean, z=" + str(heights[i]))
            ax[i, 1].imshow(var_pred[i], origin="lower", extent=(0, xdim, 0, ydim))
            ax[i, 1].set_xlabel('X-coordinate')
            ax[i, 1].set_ylabel('Y-coordinate')
            ax[i, 1].set_title("Pred Var")
        plt.show()

    # Get a plume intersection of points
    x = np.linspace(0, xdim, xdim)
    y = np.zeros_like(x)
    mean = np.zeros((zdim, xdim))
    var = np.zeros((zdim, xdim))
    for i, zh in enumerate(np.linspace(0, 50, zdim)):
        print(zh)
        height = np.ones_like(x) * zh
        mq, vq = mtt.get_prediction(t=0.0,
                                    loc=(x, y, height),
                                    num_samples=iter,
                                    from_cache=False)
        mean[i, :] = mq
        var[i, :] = vq
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(mean, origin="lower", extent=(0, xdim, 0, zdim))
    ax[0].xlabel('X-coordinate')
    ax[0].ylabel('Z-coordinate')
    ax[0].title('Mean Pred')
    ax[1].imshow(var/mean**2, origin="lower", extent=(0, xdim, 0, zdim))
    ax[1].xlabel('X-coordinate')
    ax[1].ylabel('Z-coordinate')
    ax[1].title('Var Pred')
    plt.show()

####################
# Crossflow Model
####################
if RUN_MULTIPLUME is True:
    print("Running Multiplume Example...")
    s = np.linspace(0, 1000, 100)
    extent = Extent(xrange=(0, xdim),
                    xres=xdim,
                    yrange=(0, ydim),
                    yres=ydim,
                    zrange=(0, 50),
                    zres=zdim)
    mtt1 = Crossflow(plume_loc=(0, 0, 0), extent=extent, s=s,
                     tprof=tprof, sprof=sprof,
                     rhoprof=rhoprof, vex=v0_param, area=a0_param,
                     density=rho0, curfunc=curmag, headfunc=curhead,
                     salt=s0, temp=t0, E=(alph_param, bet_param))
    mtt2 = Crossflow(plume_loc=(75, 0, 0), extent=extent, s=s,
                     tprof=tprof, sprof=sprof,
                     rhoprof=rhoprof, vex=v0_param, area=a0_param,
                     density=rho0, curfunc=curmag, headfunc=curhead,
                     salt=s0, temp=t0, E=(alph_param, bet_param))
    mtt = Multimodel([mtt1, mtt2])

    # Get a plume intersection of points
    x = np.linspace(-50, 200, 500)
    y = np.zeros_like(x)
    height = np.ones_like(x) * 35.
    pq = mtt.get_value(t=0.0, loc=(x, y, height))
    plt.plot(x, pq, c='r')
    for i in range(iter):
        mq, vq = mtt.get_prediction(t=0.0, loc=(x, y, height), num_samples=1)
        plt.plot(x, mq, c='k', alpha=0.1)
    plt.xlabel('X-coordinate')
    plt.ylabel('Prob Value')
    plt.title('Environment Slice at Y=0')
    plt.show()

    # Now get snapshot
    mean_snap = mtt.get_snapshot(t=0,
                                 z=[35.],
                                 xrange=(0, xdim),
                                 yrange=(0, ydim),
                                 xres=xdim,
                                 yres=ydim,
                                 from_cache=False)

    mean_pred, var_pred = mtt.get_snapshot_prediction(t=0,
                                                      z=[35.],
                                                      xrange=(0, xdim),
                                                      yrange=(0, ydim),
                                                      xres=xdim,
                                                      yres=ydim,
                                                      from_cache=False,
                                                      num_samples=iter)

    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    ax[0].imshow(mean_snap[0], origin="lower", extent=(0, xdim, 0, ydim))
    ax[0].set_xlabel('X-coordinate')
    ax[0].set_ylabel('Y-coordinate')
    ax[0].set_title("Snapshot")
    ax[1].imshow(mean_pred[0], origin="lower", extent=(0, xdim, 0, ydim))
    ax[1].set_xlabel('X-coordinate')
    ax[1].set_ylabel('Y-coordinate')
    ax[1].set_title("Pred Mean")
    ax[2].imshow(var_pred[0], origin="lower", extent=(0, xdim, 0, ydim))
    ax[2].set_xlabel('X-coordinate')
    ax[2].set_ylabel('Y-coordinate')
    ax[2].set_title("Pred Var")
    plt.show()

    if MAKE_CACHES is True:
        print("Writing cache...")
        mtt.write_prediction_cache(tvec=[0],
                                   xrange=(0, xdim),
                                   yrange=(0, ydim),
                                   zrange=(0, zdim),
                                   xres=xdim,
                                   yres=ydim,
                                   zres=zdim,
                                   overwrite=True,
                                   num_samples=iter)

        print("Reading from cache...")
        mean_pred, var_pred = mtt.get_snapshot_prediction(t=0,
                                                          z=[0, 10, 20, 30, 40, 50],
                                                          xrange=(0, xdim),
                                                          yrange=(0, ydim),
                                                          xres=xdim,
                                                          yres=ydim,
                                                          from_cache=True,
                                                          num_samples=iter)

        fig, ax = plt.subplots(6, 3, sharex=True, sharey=True)
        heights = [0, 10, 20, 30, 40, 50]
        for i in range(6):
            ax[i, 0].imshow(mean_pred[i], origin="lower", extent=(0, xdim, 0, ydim))
            ax[i, 0].set_xlabel('X-coordinate')
            ax[i, 0].set_ylabel('Y-coordinate')
            ax[i, 0].set_title("Pred Mean, z=" + str(heights[i]))
            ax[i, 1].imshow(var_pred[i], origin="lower", extent=(0, xdim, 0, ydim))
            ax[i, 1].set_xlabel('X-coordinate')
            ax[i, 1].set_ylabel('Y-coordinate')
            ax[i, 1].set_title("Pred Var")
        plt.show()

        # Get a plume intersection of points
        x = np.linspace(0, xdim, xdim)
        y = np.zeros_like(x)
        mean = np.zeros((zdim, xdim))
        var = np.zeros((zdim, xdim))
        for i, zh in enumerate(np.linspace(0, zdim, zdim)):
            print(zh)
            height = np.ones((zdim,)) * zh
            mq, vq = mtt.get_prediction(t=0.0,
                                        loc=(x, y, height),
                                        num_samples=iter,
                                        from_cache=True)
            mean[i, :] = mq
            var[i, :] = vq
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
        ax[0].imshow(mean, origin="lower", extent=(0, xdim, 0, zdim))
        ax[0].xlabel('X-coordinate')
        ax[0].ylabel('Z-coordinate')
        ax[0].title('Mean Pred')
        ax[1].imshow(var, origin="lower", extent=(0, xdim, 0, zdim))
        ax[1].xlabel('X-coordinate')
        ax[1].ylabel('Z-coordinate')
        ax[1].title('Var Pred')
        plt.show()
