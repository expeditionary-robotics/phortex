"""Creates multiple plume simulations under different crossflow conditions.

Allows us to estimate how far plumes may drift under different regimes
and standard initial conditions."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import utm
import os

from fumes.environment.mtt import CrossflowMTT
from fumes.environment.utils import eos_rho
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile


def create_crossflow_worlds(locs, extent, s, tprof, sprof, rhoprof, curfuncs, headfuncs, v0s, a0s, s0s, t0s, rho0s, lams, entrainments):
    """Creates a list of crossflow worlds."""
    envs = []
    for loc, v0, a0, s0, t0, rho0, curfunc, headfunc, lam, entrainment in zip(locs, v0s, a0s, s0s, t0s, rho0s, curfuncs, headfuncs, lams, entrainments):
        w = CrossflowMTT(plume_loc=loc,
                         extent=extent,
                         s=s,
                         tprof=tprof,
                         sprof=sprof,
                         rhoprof=rhoprof,
                         curfunc=curfunc,
                         headfunc=headfunc,
                         vex=v0,
                         area=a0,
                         salt=s0,
                         temp=t0,
                         density=rho0,
                         lam=lam,
                         entrainment=entrainment)
        envs.append(w)
    return envs


# Globals
SKIP = 100
REFERENCE = (9.9055599, -104.294361, -2555.)
RX, RY, RN, RL = utm.from_latlon(REFERENCE[0], REFERENCE[1])
# SMOKING_CHIMNEYS = [(9.905665, -104.294522, -2556.349),
#                     (9.905230, -104.294360, -2550.3),
#                     (9.905639, -104.294531, -2555.558)]
SMOKING_CHIMNEYS = [(9.905665, -104.294522, -2555),
                    (9.905665, -104.294522, -2555),
                    (9.905665, -104.294522, -2555)]

# Parameters
z = np.linspace(0, 300, 100)  # height to integrate over
s = np.linspace(0, 200, 200)  # length to integrate over

# Profiles
casts_df = pd.read_csv(os.path.join(os.getenv("EPR_DATA"), "ctd/proc/cast_training_data.csv"))
datax = 2555. - casts_df.depth
datayt = casts_df.pot_temp
datays = casts_df.salinity
tprof_obj = Profile(datax[::SKIP], datayt[::SKIP], training_iter=30, learning_rate=0.1)
tprof = tprof_obj.profile
sprof_obj = Profile(datax[::SKIP], datays[::SKIP], training_iter=30, learning_rate=0.1)
sprof = sprof_obj.profile
plt.plot(sprof(np.linspace(0, 1000, 100)), np.linspace(0, 1000, 100))
plt.show()
rhoprof = eos_rho  # function that computes density as func of S, T

times = np.linspace(0, 24 * 3600, 24 * 3600)
query_times = [0]  # , 6, 12]  # in hours
def headfunc(t): return 0  # headfunc(times)
def curfunc(z, t): return 0.12  # curfunc(None, times)


# location of the vents; known
locs = []
for sc in SMOKING_CHIMNEYS:
    cx, cy, _, _ = utm.from_latlon(sc[0], sc[1])
    locs.append((cx - RX, cy - RY, sc[2] - REFERENCE[2]))
print(locs)

# fluid exit velocity; unknown
v0s = [0.05, 0.05, 0.05]

# orifice area; unknown
a0s = [0.14, 0.14, 0.14]

# source salinity; estimate
s0s = [34.608, 34.608, 34.608]

# source temperature; estimate
t0s = [370, 370, 370]

# source density
rho0s = [700, 700, 700]  # source density

# assume they are all essentially symmetric
lams = [1.0, 1.0, 1.0]  # for crossflow, major-minor axis ratio

# assume that the mixing coefficients are fixed for all plumes
entrainments = [[0.12, 0.25], [0.12, 0.25], [0.12, 0.25]]  # entrainment coeffs

# current functions
curfuncs = [lambda x, t: 0.05, lambda x, t: 0.1, lambda x, t: 0.15]
headfuncs = [headfunc, headfunc, headfunc]

# set the simulation boundaries
extent = Extent(xrange=(-100., 100.),
                xres=100,
                yrange=(-100., 100.),
                yres=100,
                zrange=(0, 100),
                zres=10,
                global_origin=(0, 0, 0))

# define the environments
envs = create_crossflow_worlds(locs=locs,
                              extent=extent,
                              s=s,
                              tprof=tprof,
                              sprof=sprof,
                              rhoprof=rhoprof,
                              curfuncs=curfuncs,
                              headfuncs=headfuncs,
                              v0s=v0s,
                              a0s=a0s,
                              s0s=s0s,
                              t0s=t0s,
                              rho0s=rho0s,
                              lams=lams,
                              entrainments=entrainments)

# Get a plume intersection
plume_colors = ["r", "b", "g"]
currents = [g(0, 0) for g in curfuncs]
for current, c, plume in zip(currents, plume_colors, envs):
    le, cl, re = plume.envelope(t=0.0)
    lefill = np.interp(re[0], le[0], le[1])
    plt.fill_between(re[0], re[1], lefill, interpolate=True, alpha=0.1, color=c, label=f"{current} m/s")
    plt.plot(*cl, c=c, ls="--")
    plt.plot(*le, c=c, alpha=0.5)
    plt.plot(*re, c=c, alpha=0.5)
plt.hlines([80.], [0.], [200.], colors=["k"], label="Sentry Survey Height")
plt.title("Plume Envelope")
plt.xlabel("X in crossflow direction (meters)")
plt.ylabel("Z (meters)")
plt.legend()
plt.show()

