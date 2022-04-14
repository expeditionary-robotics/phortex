"""Draws samples of unknown parameters and computes sensitivity in model."""

import numpy as np
import matplotlib.pyplot as plt
from fumes.environment.mtt import StationaryMTT
from fumes.model.mtt import MTT
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S

# Parameters
z = np.linspace(0, 200, 100)  # height to integrate over
tprof = pacific_sp_T  # function that describes background temp
sprof = pacific_sp_S  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T
v0 = 0.4  # source exit velocity
a0 = 0.1  # source area
s0 = 34.608  # source salinity
t0 = 300  # source temperature
rho0 = eos_rho(t0, s0)  # source density

xdim = 200  # target simulation xdim
ydim = 200  # target simulation ydim
zdim = 200
resolution = 10  # number of voxels in each dimension
iter = 500  # number of samples to search over

# Create Environment
env = StationaryMTT(plume_loc=(0, 0), z=z, tprof=tprof, sprof=sprof, rhoprof=rhoprof,
                    vex=v0, area=a0, density=rho0, salt=s0, temp=t0)

# Sample velocity and observe environment output distributions
v0 = np.random.uniform(size=iter)
vert = []
salt = []
temp = []
for i in range(iter):
    env.v0 = v0[i]
    env.solve(t=0.)
    V, S, T, P = env.get_value(0, (0, 0, 50), return_all=True)
    vert.append(V)
    salt.append(S)
    temp.append(T)

plt.hist(vert, bins=50)
plt.xlabel("Vertical Velocity")
plt.ylabel("Count")
plt.title("Exit Velocity")
plt.show()
plt.show()

# Sample entrainment and observe environment output distributions
E = np.random.uniform(size=iter)
vert = []
salt = []
temp = []
for i in range(iter):
    env.entrainment = E[i]
    env.solve(t=0.)
    V, S, T, P = env.get_value(0, (0, 0, 50), return_all=True)
    vert.append(V)
    salt.append(S)
    temp.append(T)

plt.hist(vert, bins=50)
plt.xlabel("Vertical Velocity")
plt.ylabel("Count")
plt.title("Entrainment")
plt.show()

# Look at the multiplicative effects
E = np.random.uniform(size=iter)
v0 = np.random.uniform(size=iter)
vert = []
salt = []
temp = []
for i in range(100):
    for j in range(100):
        env.entrainment = E[i]
        env.v0 = v0[j]
        env.solve(t=0.)
        V, S, T, P = env.get_value(0, (0, 0, 50), return_all=True)
        vert.append(V)
        salt.append(S)
        temp.append(T)

plt.hist(vert, bins=50)
plt.xlabel("Vertical Velocity")
plt.ylabel("Count")
plt.title("Entrainment and Exit Velocity")
plt.show()
