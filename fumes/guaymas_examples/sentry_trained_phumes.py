"""Environment estimated by the trained PHUMES model on at-sea Sentry data.
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


# "Global" Model Parameters
s = np.linspace(0, 1000, 200)  # distance to integrate over
z = np.linspace(0, 200, 100)  # height to integrate over
tprof = Profile(z, pacific_sp_T(z))  # function that describes background temp
sprof = Profile(z, pacific_sp_S(z))  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T

# True Source Params
v0 = 0.58
a0 = 0.82
s0 = 34.908  # source salinity
t0 = 340  # source temperature
rho0 = eos_rho(t0, s0)  # source density
E = (0.15, 0.19)

# Model Simulation Params
extent = Extent(xrange=(0., 1000.),
                xres=100,
                yrange=(0., 1000.),
                yres=100,
                zrange=(0, 200),
                zres=50,
                global_origin=(0., 0., -1850.))

# Robot params
vel = 0.5  # robot velocity (in meters/second)
com_window = 120  # communication window (in seconds)
altitude = 70.0  # flight altitude (in meters)

# Create Environment
print("Creating learned environment...")
env = CrossflowMTT(plume_loc=(0, 0, -1850.), extent=extent, s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0, area=a0, entrainment=E)

# plot the envelope for a set of currents
T = [0]
print(curfunc(None, 0.0))
for t in T:
    le, cl, re = env.envelope(t=t)

    plt.plot(*cl, label="Centerline",c='b',ls='--')
    plt.plot(*le, label="Left Extent", c='b')
    plt.plot(*re, label="Right Extent", c='b')
    lefill = np.interp(re[0], le[0], le[1])
    plt.fill_between(re[0], re[1], lefill, interpolate=True, alpha=0.1)
plt.vlines([100, 600], ymin=[-1850, -1850], ymax=[-1550, -1550],
           colors=['r', 'r'])
plt.title("Plume Envelope")
plt.xlabel("X in crossflow direction (meters)")
plt.ylabel("Z (meters)")
plt.show()
