"""Draws samples from a Crossflow environment to update a Crossflow model."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from distfit import distfit
from fumes.environment.mtt import CrossflowMTT
from fumes.environment.profile import Profile
from fumes.environment.current import CurrMag, CurrHead
from fumes.model.mtt import Crossflow
from fumes.model.parameter import Parameter
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.environment import Extent


def _get_plotting_polygon(t, mod, color):
    le, _, re = mod.envelope(t=t)
    x1, z1 = le
    x2, z2 = re
    verts = [*zip(x1, z1), *zip(reversed(x2), reversed(z2))]
    poly = Polygon(verts, facecolor=color, edgecolor=color, alpha=0.2)
    return poly

# Simulation Params
xdim = 800  # target simulation xdim
ydim = 800  # target simulation ydim
zdim = 200  # target simulation zdim
T = np.linspace(0, 12 * 3600, 10)  # time snapshots
resolution = 10  # number of voxels in each dimension
iterations = 100  # number of samples to search over
burn = 10  # number of samples for burn-in
thresh = 1e-15  # probability threshold for a detection

# "Global" Parameters
s = np.linspace(0, 500, 100)  # distance to integrate over
z = np.linspace(0, 200, 100)  # heights to consider
tprof = Profile(z, pacific_sp_T(z))  # function that describes background temp
sprof = Profile(z, pacific_sp_S(z))  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T
extent = Extent(xrange=(0., xdim.), xres=xdim, yrange=(0., ydim), yres=ydim)


# True Source Params
v0 = 0.4  # source exit velocity
a0 = 0.1  # source area
s0 = 34.608  # source salinity
t0 = 300  # source temperature
rho0 = eos_rho(t0, s0)  # source density
E = (0.12, 0.1)

# Inferred Source Params
v0_inf = distfit(distr='uniform')
v0_inf.fit_transform(np.random.uniform(0.05, 1.5, 2000))
v0_prop = sp.stats.norm(loc=0, scale=0.1)
v0_param = Parameter(v0_inf, v0_prop)

a0_inf = distfit(distr='uniform')
a0_inf.fit_transform(np.random.uniform(0.05, 0.5, 2000))
a0_prop = sp.stats.norm(loc=0, scale=0.1)
a0_param = Parameter(a0_inf, a0_prop)

alph_inf = distfit(distr='uniform')
alph_inf.fit_transform(np.random.uniform(0.1, 0.2, 2000))
alph_prop = sp.stats.norm(loc=0, scale=0.01)
alph_param = Parameter(alph_inf, alph_prop)

bet_inf = distfit(distr='uniform')
bet_inf.fit_transform(np.random.uniform(0.05, 0.25, 2000))
bet_prop = sp.stats.norm(loc=0, scale=0.05)
bet_param = Parameter(bet_inf, bet_prop)

# Current params
t = np.linspace(0, 12*3600, 5000)
curmag = CurrMag(t, curfunc(None, t), training_iter=100, learning_rate=0.1)
curhead = CurrHead(t, headfunc(t)+np.random.normal(0, 0.1, t.shape), training_iter=100, learning_rate=0.1)

# Create Environment
env = CrossflowMTT(plume_loc=(0, 0, 0), s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0, area=a0, entrainment=E)


# Create Model
mtt = Crossflow(extent=extent, plume_loc=(0, 0, 0), s=s, tprof=tprof, sprof=sprof,
                rhoprof=rhoprof, vex=v0_param, area=a0_param, density=rho0,
                curfunc=curmag, headfunc=curhead,
                salt=s0, temp=t0, E=(alph_param, bet_param))

# Sample Environment
x, y, z = np.meshgrid(np.linspace(-xdim / 2., xdim / 2., resolution),
                      np.linspace(-ydim / 2., ydim / 2., resolution),
                      np.linspace(0, zdim, resolution))
x, y, z = x.flatten(), y.flatten(), z.flatten()

class_samps = np.zeros((T.shape[0], resolution * resolution * resolution))
locs = np.zeros((T.shape[0], 3, resolution**3))
for i, t in enumerate(T):
    temp = env.get_value(t, (x, y, z))
    class_samps[i, :] = temp > thresh
    locs[i, :, :] = (x, y, z)

    print("Proportion of in-plume samples: ",
          np.sum(temp > thresh) / len(temp))

# Set arbitrary plotting frame
plot_t = T[1]

# Show the generating environment envelope and
# samples for an arbitrary frame
env.solve(t=plot_t)
plt.plot(env.x_disp(t=plot_t), env.z_disp(t=plot_t),
         alpha=0.2, c='b', label='Generating Environment')
poly = _get_plotting_polygon(plot_t, env, 'b')
plt.gca().add_patch(poly)
plt.scatter(x, z, label='Samples')

# Show the starting model envelope
mtt.solve(t=plot_t)
plt.plot(mtt.odesys.x_disp(t=plot_t), mtt.odesys.z_disp(t=plot_t),
         alpha=0.2, c='g',
         label='Initialized Model')
poly = _get_plotting_polygon(plot_t, mtt.odesys, 'g')
plt.gca().add_patch(poly)

# Bulk MTT Model Update
print("Starting multiple real update")
print("Multiple real observations: ", mtt.update(T,
                                                 locs,
                                                 class_samps,
                                                 num_samps=iterations,
                                                 burnin=burn,
                                                 thresh=thresh))
# Show the updated model envelope
mtt.solve(t=plot_t)
plt.plot(mtt.odesys.x_disp(t=plot_t), mtt.odesys.z_disp(t=plot_t),
         alpha=0.2, c='k', label='Multiple Update')
poly = _get_plotting_polygon(plot_t, mtt.odesys, 'k')
plt.gca().add_patch(poly)

plt.title("Plume Envelope")
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.legend()
plt.show()
