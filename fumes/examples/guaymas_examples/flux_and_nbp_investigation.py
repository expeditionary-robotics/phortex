"""Investigation of Flux and Neutrally-Buoyant Height"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fumes.environment.mtt import CrossflowMTT
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile
from fumes.environment.current import CurrHead, CurrMag

from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, curfunc, headfunc
from fumes.utils.save_mission import load_experiment_json

def compute_NBH_height_points(area, velocity, env, t, ax, height):
    nbh = np.zeros_like(area)
    print(t, height)
    for i, a in enumerate(area[0, :]):
        for j, v in enumerate(velocity[:, 0]):
            env.v0 = v
            env.a0 = a
            env.solve(t=t, overwrite=True)
            _, c, _ = env.envelope(t=t)
            nbh[j, i] = np.nanmean(c[1][100:])
    nbhp = ax.contour(areax, velocityx, nbh, 10)
    ax.clabel(nbhp, inline=1, fontsize=10)
    nbhs = ax.contour(areax, velocityx, nbh, levels=height)
    return nbhs


SAVE_DIRECTORY = os.path.join(os.getenv("FUMES_OUTPUT"), f"investigation")
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
DATA_HOME = "/home/vpreston/rrg/src/planning/sentry/data/"
TEMPERATURE_BKGND_FILENAME = os.path.join(DATA_HOME, "profiles/proc_temp_profile.csv")
SALINITY_BKGND_FILENAME = os.path.join(DATA_HOME, "profiles/proc_salt_profile.csv")
CURRENT_FILENAME = os.path.join(DATA_HOME, "currents/proc_current_train_profile1.csv")
REFERENCE = (float(os.getenv("LAT")),
             float(os.getenv("LON")),
             float(os.getenv("DEP")))
SR = 30  # subsample rate
TQuery = np.linspace(0, 24*3600, 4)
HQuery = [(-1730, -1680), (-1600, -1530)]
Color = [("red", "pink"), ("gray", "yellow")]

temp = pd.read_csv(TEMPERATURE_BKGND_FILENAME)
datax_temp, datay_temp = temp['depth'], temp['temperature']
print("Training temperature...")
TPROF = Profile(datax_temp[::SR], datay_temp[::SR], training_iter=100, learning_rate=0.1)

salt = pd.read_csv(SALINITY_BKGND_FILENAME)
datax_salt, datay_salt = salt['depth'], salt['salinity']
print("Training salinity...")
SPROF = Profile(datax_salt[::SR], datay_salt[::SR], training_iter=100, learning_rate=0.1)

# current = pd.read_csv(CURRENT_FILENAME)
# current = current.dropna()
# datax_cur, datay_mag, datay_head = current['hours'], current['mag_mps_train'], current['head_rad_train']
print("Training current magnitude...")
# CURRMAG = CurrMag(datax_cur, datay_mag, training_iter=100, learning_rate=0.5)
current = lambda m, x: x / (3600.*24) * 0.1 + 0.03  # just get the ranges we care about

print("Training current heading...")
heading = lambda x: 0  # this doesn't matter for these plots, just set to something
# CURRHEAD = CurrHead(datax_cur, datay_head * 180 / np.pi, training_iter=200, learning_rate=0.1)

EXTENT = Extent(xrange=(-100., 500.),
                xres=10,
                yrange=(-100., 500.),
                yres=10,
                zrange=(0, 200),
                zres=2,
                global_origin=(0., 0., -1850.))

print("Creating environments...")
BASELINE = CrossflowMTT(extent=EXTENT,
                        plume_loc=(0., 0., -1850.),
                        s=np.linspace(0, 1000, 200),
                        curfunc=current,
                        headfunc=heading,
                        tprof=TPROF.profile,
                        sprof=SPROF.profile,
                        rhoprof=eos_rho,
                        vex=0.58,
                        area=0.82,
                        density=eos_rho(340, 34.908),
                        salt=34.908,
                        temp=340,
                        lam=1.0,
                        entrainment=[0.3, 0.3])

env = CrossflowMTT(extent=EXTENT,
                   plume_loc=(0., 0., -1850.),
                   s=np.linspace(0, 1000, 200),
                   curfunc=current,
                   headfunc=heading,
                   tprof=TPROF.profile,
                   sprof=SPROF.profile,
                   rhoprof=eos_rho,
                   vex=0.58,
                   area=0.82,
                   density=eos_rho(340, 34.908),
                   salt=34.908,
                   temp=340,
                   lam=1.0,
                   entrainment=[0.3, 0.3])

if __name__ == "__main__":
    # compute flux contour lines
    areax, velocityx = np.meshgrid(np.linspace(0.05, 2.0, 10),
                                   np.linspace(0.05, 2.0, 10))
    q0 = (areax / np.pi * velocityx)
    m0 = (q0 * velocityx)
    f0 = 9.81 * 10**(-4) * (340 - TPROF.profile(-1850.)) * q0

    fig, ax = plt.subplots(1, 7, sharex=True, sharey=True, figsize=(15, 8))
    qs = ax[0].contour(areax, velocityx, q0, 10)
    # qst = ax[0].contour(areax, velocityx, q0, levels=[BASELINE.q0])
    ax[0].clabel(qs, inline=1, fontsize=10)
    # ax[0].clabel(qst, inline=1, fontsize=10)
    ax[0].scatter(BASELINE.a0, BASELINE.v0, color="green")
    ax[0].set_xlabel("Area")
    ax[0].set_ylabel("Velocity")
    ax[0].set_title("Heat Flux")

    ms = ax[1].contour(areax, velocityx, m0, 10)
    # mst = ax[1].contour(areax, velocityx, m0, levels=[BASELINE.m0])
    ax[1].clabel(ms, inline=1, fontsize=10)
    # ax[1].clabel(mst, inline=1, fontsize=10)
    ax[1].scatter(BASELINE.a0, BASELINE.v0, color="green")
    ax[1].set_xlabel("Area")
    ax[1].set_title("Momentum Flux")

    bs = ax[2].contour(areax, velocityx, f0, 10)
    # bst = ax[2].contour(areax, velocityx, f0, levels=[BASELINE.f0])
    ax[2].clabel(bs, inline=1, fontsize=10)
    # ax[2].clabel(bst, inline=1, fontsize=10)
    ax[2].scatter(BASELINE.a0, BASELINE.v0, color="green")
    ax[2].set_xlabel("Area")
    ax[2].set_title("Buoyancy Flux")

    # compute the NBH for multiple currents
    for m, t in enumerate(TQuery):
        for h, c in zip(HQuery, Color):
            nbhs = compute_NBH_height_points(areax, velocityx, env, t=t, height=h, ax=ax[m+3])
            ax[m+3].scatter(BASELINE.a0, BASELINE.v0, color="green")
            for hs in range(len(h)):
                try:
                    p = nbhs.collections[hs].get_paths()[0]
                    v = p.vertices
                    x = v[:, 0]
                    y = v[:, 1]
                    ax[m+3].scatter(x, y, color=c[hs])  # plot the NBH curve
                    ax[2].scatter(x, y, color=c[hs])  # plot the NBH curve
                    ax[1].scatter(x, y, color=c[hs])  # plot the NBH curve
                    ax[0].scatter(x, y, color=c[hs])  # plot the NBH curve
                except:
                    pass
        ax[m+3].set_xlabel("Area")
        ax[m+3].set_title(f"NBP Height {current(None, t)}")
    ax[3].scatter(BASELINE.a0, BASELINE.v0, color="green")
    ax[3].set_xlabel("Area")
    ax[3].set_title("Neutrally-Buoyant Plume Height")

    plt.show()
    fig.savefig(os.path.join(SAVE_DIRECTORY, f"fnbh_area_vel_03.svg"))
    plt.close()

    # compute nbh with alpha beta
    # alphax, betax = np.meshgrid(np.linspace(0.05, 0.25, 12), np.linspace(0.05, 0.25, 10))
    # print(alphax.shape)
    # fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(15, 8))
    # nbh = np.zeros_like(alphax)
    # for i, a in enumerate(alphax[0, :]):
    #     for j, b in enumerate(betax[:, 0]):
    #         print(a, b)
    #         env.v0 = 0.58
    #         env.a0 = 0.82
    #         env.entrainment[0] = a
    #         env.entrainment[1] = b
    #         env.solve(t=0, overwrite=True)
    #         _, c, _ = env.envelope(t=0)
    #         nbh[j, i] = np.nanmean(c[1][100:])
    #         print(np.nanmean(c[1][100:]))
    # nbh = ax.contour(alphax, betax, nbh, 10)
    # ax.clabel(nbh, inline=1, fontsize=10)
    # ax.scatter(BASELINE.entrainment[0], BASELINE.entrainment[1], color="green")
    # ax.set_xlabel("Alpha")
    # ax.set_ylabel("Beta")
    # ax.set_title("Neutrally-Buoyant Plume Height")

    # plt.show()
    # fig.savefig(os.path.join(SAVE_DIRECTORY, f"fnbh_alpha_beta.svg"))
    # plt.close()

    # compute nbh over time
    # T = np.linspace(0, 3600 * 24, 50)
    # C = np.zeros_like(T)
    # for i, t in enumerate(T):
    #     _, c, _ = BASELINE.envelope(t=t)
    #     C[i] = np.nanmean(c[1][100:])
    # fig, ax = plt.subplots(2, 1, sharex=True, sharey=False)
    # ax[0].plot(T, C)
    # ax[1].plot(T, CURRMAG.magnitude(None, T))
    # ax[1].set_xlabel("Time")
    # ax[1].set_ylabel("Current Magnitude")
    # ax[0].set_ylabel("NBH")
    # plt.show()
    # fig.savefig(os.path.join(SAVE_DIRECTORY, f"nbh_current.svg"))
    # plt.close()

    # plt.hist(C, bins=10)
    # plt.show()
