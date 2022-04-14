"""Draws samples from a Multiplume environment to update a Multiplume model."""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from distfit import distfit
from fumes.environment.mtt import CrossflowMTT, Multiplume
from fumes.environment.profile import Profile
from fumes.environment.current import CurrMag, CurrHead
from fumes.model.mtt import Crossflow, Multimodel
from fumes.model.parameter import Parameter
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.environment import Extent


def _get_plotting_polygon(t, mod, color):
    le, _, re = mod.envelope(t=t)
    xloc = mod.loc[0]
    x1, z1 = le
    x2, z2 = re
    verts = [*zip(x1+xloc, z1), *zip(reversed(x2+xloc), reversed(z2))]
    poly = Polygon(verts, facecolor=color, edgecolor=color, alpha=0.1)
    return poly


# "Global" Parameters
s = np.linspace(0, 500, 100)  # distance to integrate over
z = np.linspace(0, 300, 300)  # relevant altitudes to consider
tprof = Profile(z, pacific_sp_T(z))  # function that describes background temp
sprof = Profile(z, pacific_sp_S (z)) # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T

# Environment params
extent = Extent(xrange=(0., 500.), xres=100, yrange=(0., 500.), yres=100)

# True Source Params (let both plumes be the same)
v0 = 0.4  # source exit velocity
a0 = 0.1  # source area
s0 = 34.608  # source salinity
t0 = 300  # source temperature
rho0 = eos_rho(t0, s0)  # source density
E = (0.12, 0.1)

# Inferred Source Params (let both plumes be seeded similarly)
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
t = np.linspace(0, 12*3600, 1500)
curmag = CurrMag(t, curfunc(None, t), training_iter=100, learning_rate=0.1)
curhead = CurrHead(t, headfunc(t)+np.random.normal(0, 0.1, t.shape), training_iter=100, learning_rate=0.1)

# Simulation Params
xdim = 1000  # target simulation xdim
ydim = 1000  # target simulation ydim
zdim = 200  # target simulation zdim
T = np.linspace(0, 12*3600, 6)  # time snapshots
resolution = 10  # number of voxels in each dimension
iter = 100  # number of samples to search over
burn = 10  # number of samples for burn-in
thresh = 1e-15  # probability threshold for a detection

# Create Environment
cf1 = CrossflowMTT(plume_loc=(0, 0, 0), s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0, area=a0, entrainment=E)

cf2 = CrossflowMTT(plume_loc=(200, 0, 0), s=s,
                   tprof=pacific_sp_T, sprof=pacific_sp_S, rhoprof=rhoprof,
                   density=rho0, salt=s0, temp=t0,
                   curfunc=curfunc, headfunc=headfunc,
                   vex=v0, area=a0, entrainment=E)

env = Multiplume([cf1, cf2])

# Create Model
mc1 = Crossflow(extent=extent, plume_loc=(0, 0, 0), s=s, tprof=tprof, sprof=sprof,
                rhoprof=rhoprof, vex=v0_param, area=a0_param, density=rho0,
                curfunc=curmag, headfunc=curhead,
                salt=s0, temp=t0, E=(alph_param, bet_param))

mc2 = Crossflow(extent=extent, plume_loc=(200, 0, 0), s=s, tprof=tprof, sprof=sprof,
                rhoprof=rhoprof, vex=v0_param, area=a0_param, density=rho0,
                curfunc=curmag, headfunc=curhead,
                salt=s0, temp=t0, E=(alph_param, bet_param))

mtt = Multimodel(multiplume_models=[mc1, mc2])

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

# Show the starting environment envelope + samples
plot_t = T[1]
env.solve(t=plot_t)
plt.plot(env.models[0].x_disp(plot_t) + env.origins[0][0],
         env.models[0].z_disp(plot_t),
         alpha=0.2, c='b', label='Generating Environment')
plt.plot(env.models[1].x_disp(plot_t) + env.origins[1][0],
         env.models[1].z_disp(plot_t),
         alpha=0.2, c='b')
poly1 = _get_plotting_polygon(plot_t, env.models[0], 'b')
poly2 = _get_plotting_polygon(plot_t, env.models[1], 'b')
plt.gca().add_patch(poly1)
plt.gca().add_patch(poly2)
plt.scatter(x, z, label='Samples')

# Show the starting model envelope
mtt.solve(t=plot_t)
plt.plot(mtt.models[0].odesys.x_disp(plot_t) + mtt.models[0].loc[0],
         mtt.models[0].odesys.z_disp(plot_t),
         alpha=0.2, c='g',
         label='Initialized Model')
plt.plot(mtt.models[1].odesys.x_disp(plot_t) + mtt.models[1].loc[0],
         mtt.models[1].odesys.z_disp(plot_t),
         alpha=0.2, c='g')
poly1 = _get_plotting_polygon(plot_t, mtt.models[0].odesys, 'g')
poly2 = _get_plotting_polygon(plot_t, mtt.models[1].odesys, 'g')
plt.gca().add_patch(poly1)
plt.gca().add_patch(poly2)

# Bulk MTT Model Update
print("Starting multiple real update")
print("Multiple real observations: ", mtt.update(T,
                                                 locs,
                                                 class_samps,
                                                 num_samps=iter,
                                                 burnin=burn,
                                                 thresh=thresh))
# Show the updated model envelope
plt.plot(mtt.models[0].odesys.x_disp(plot_t) + mtt.models[0].loc[0],
         mtt.models[0].odesys.z_disp(plot_t),
         alpha=0.2, c='k',
         label='Multiple Update')
plt.plot(mtt.models[1].odesys.x_disp(plot_t) + mtt.models[1].loc[0],
         mtt.models[1].odesys.z_disp(plot_t),
         alpha=0.2, c='k')
poly1 = _get_plotting_polygon(plot_t, mtt.models[0].odesys, 'k')
poly2 = _get_plotting_polygon(plot_t, mtt.models[1].odesys, 'k')
plt.gca().add_patch(poly1)
plt.gca().add_patch(poly2)

plt.title("Plume Envelope")
plt.xlabel("X (meters)")
plt.ylabel("Z (meters)")
plt.legend()
plt.show()
