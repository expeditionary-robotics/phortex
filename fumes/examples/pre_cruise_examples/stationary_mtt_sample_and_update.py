"""Draws samples from a fixed MTT environment to update a MTT model."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from distfit import distfit
from fumes.environment.mtt import StationaryMTT
from fumes.environment.profile import Profile
from fumes.model.mtt import MTT
from fumes.model.parameter import Parameter
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S

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

# Simulation Params
xdim = 200  # target simulation xdim
ydim = 200  # target simulation ydim
zdim = 200
resolution = 10  # number of voxels in each dimension
iter = 100  # number of samples to search over
burn = 50  # number of samples for burn-in
thresh = 1e-5  # probability threshold for a detection

# Create Environment
env = StationaryMTT(plume_loc=(0, 0, 0), z=z,
                    tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                    density=rho0, salt=s0, temp=t0,
                    vex=v0, area=a0, entrainment=E)

# Create Model
mtt = MTT(plume_loc=(0, 0, 0), z=z, tprof=tprof, sprof=sprof, rhoprof=rhoprof,
          vex=v0_param, area=a0_param, density=rho0, salt=s0, temp=t0,
          E=E_param)

# Sample Environment
x, y, z = np.meshgrid(np.linspace(-xdim / 2., xdim / 2., resolution),
                      np.linspace(-ydim / 2., ydim / 2., resolution),
                      np.linspace(0, zdim, resolution))
x, y, z = x.flatten(), y.flatten(), z.flatten()

class_samps = env.get_value(0, (x, y, z)) > thresh
print("Proportion of in-plume samples: ",
      np.sum(class_samps) / class_samps.shape[0])

# Show the starting environment envelope + samples
le, cl, re = env.extent(t=0.0)
plt.fill_betweenx(env.z, le, re, alpha=0.5, label='Generating Environment')
plt.scatter(x, z, label='Samples')

# Show the starting model envelope
mle, mcl, mre = mtt.odesys.extent(t=0.0)
plt.fill_betweenx(mtt.z, mle, mre, alpha=0.5, label='Initialized Model')

# Bulk MTT Model Update
print("Starting multiple real update")
print("Multiple real observations: ", mtt.update(0,
                                                 (x, y, z),
                                                 class_samps,
                                                 num_samps=iter,
                                                 burnin=burn,
                                                 thresh=thresh))
# Show the updated model envelope
mle, mcl, mre = mtt.odesys.extent(t=0.0)
plt.fill_betweenx(mtt.z, mle, mre,
                  alpha=0.5,
                  label='Multiple Science Update Model')

plt.title("Plume Envelope")
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.legend()
plt.show()
