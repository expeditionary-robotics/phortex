"""Draws samples of unknown parameters and computes sensitivity in model."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from fumes.environment.mtt import CrossflowMTT, Multiplume
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.environment import Extent

# Parameters
s = np.linspace(0, 1000, 100)  # length to integrate over
tprof = pacific_sp_T  # function that describes background temp
sprof = pacific_sp_S  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T


loc = (0., 0.)  # source location

v0 = 0.4  # source exit velocity
a0 = 0.1  # source area
s0 = 34.608  # source salinity
t0 = 300.  # source temperature
rho0 = eos_rho(t0, s0)  # source density

lam = 0.5  # for crossflow, major-minor axis ratio
entrainment = [0.12, 0.2]  # entrainment coeffs

xdim = 200  # target simulation xdim
ydim = 200  # target simulation ydim
zdim = 200
resolution = 10  # number of voxels in each dimension
T = np.linspace(0, 5.9 * 3600, 10)  # time axis
iter = 10  # number of samples to search over


def _sort_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    sorts = sorted(zip(labels, handles))
    tups = zip(*sorts)
    labels, handles = [list(tup) for tup in tups]
    return labels, handles


# Create Environment
env = CrossflowMTT(plume_loc=loc,
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

# Examine Exit Velocity Variations
v0 = np.random.uniform(0.05, 1.5, size=iter)
fig, ax = plt.subplots(2, 1, sharex=True)

for v in v0:
    env.v0 = v
    env.q0 = env.lam * env.a0 / np.pi * env.v0  # source heat flux
    env.m0 = env.q0 * env.v0  # source momentum flux
    env.f0 = -env.g * 10**(-4) * (env.t0 - tprof(0)) * env.q0
    v0_height = []
    v0_cross = []
    for t in T:
        env.solve(t=t)
        temp = env.x_disp(t=t)[env.theta(t=t) < 0.05]
        try:
            v0_cross.append(temp[0])
        except IndexError:
            v0_cross.append(0)
        v0_height.append(np.nanmax(env.z_disp(t=t)))
    ax[0].scatter(T, v0_cross, label=np.round(v, 4))
    ax[1].scatter(T, v0_height, label=np.round(v, 4))

ax0_labels, ax0_handles = _sort_legend(ax[0])
ax1_labels, ax1_handles = _sort_legend(ax[1])
ax[0].legend(ax0_handles, ax0_labels, title="Exit Velocity, m/s")
ax[1].legend(ax1_handles, ax1_labels, title="Exit Velocity, m/s")
ax[1].set_xlabel("Time")
ax[0].set_ylabel("Crossflow Distance to Peak")
ax[1].set_ylabel("Peak Height")
plt.show()

# # Examine Area Variations
a0 = np.random.uniform(0.1, 1.0, size=iter)
fig, ax = plt.subplots(2, 1, sharex=True)

for a in a0:
    env.a0 = a
    env.q0 = env.lam * env.a0 / np.pi * env.v0  # source heat flux
    env.m0 = env.q0 * env.v0  # source momentum flux
    env.f0 = -env.g * 10**(-4) * (env.t0 - tprof(0)) * env.q0
    a0_height = []
    a0_cross = []
    for t in T:
        env.solve(t=t)
        temp = env.x_disp(t=t)[env.theta(t=t) < 0.05]
        try:
            a0_cross.append(temp[0])
        except IndexError:
            a0_cross.append(0)
        a0_height.append(np.nanmax(env.z_disp(t=t)))
    ax[0].scatter(T, a0_cross, label=np.round(a, 4))
    ax[1].scatter(T, a0_height, label=np.round(a, 4))

ax0_labels, ax0_handles = _sort_legend(ax[0])
ax1_labels, ax1_handles = _sort_legend(ax[1])
ax[0].legend(ax0_handles, ax0_labels, title="Source Area, m^2")
ax[1].legend(ax1_handles, ax1_labels, title="Source Area, m^2")
ax[1].set_xlabel("Time")
ax[0].set_ylabel("Crossflow Distance to Peak")
ax[1].set_ylabel("Peak Height")
plt.show()

# Examine Entrainment Variations
alf = np.random.uniform(0.1, 0.3, size=iter)
bet = np.random.uniform(0.2, 0.7, size=iter)
fig, ax = plt.subplots(2, 1, sharex=True)

for a, b in zip(alf, bet):
    env.entrainment = [a, b]
    E_height = []
    E_cross = []
    for t in T:
        env.solve(t=t)
        temp = env.x_disp(t=t)[env.theta(t=t) < 0.05]
        try:
            E_cross.append(temp[0])
        except IndexError:
            E_cross.append(0)
        E_height.append(np.nanmax(env.z_disp(t=t)))
    ax[0].scatter(T, E_cross, label=[np.round(a, 4), np.round(b, 4)])
    ax[1].scatter(T, E_height, label=[np.round(a, 4), np.round(b, 4)])

ax0_labels, ax0_handles = _sort_legend(ax[0])
ax1_labels, ax1_handles = _sort_legend(ax[1])
ax[0].legend(ax0_handles, ax0_labels, title="Entrainment")
ax[1].legend(ax1_handles, ax1_labels, title="Entrainment")
ax[1].set_xlabel("Time")
ax[0].set_ylabel("Crossflow Distance to Peak")
ax[1].set_ylabel("Peak Height")
plt.show()

# show overlap between multiple plumes experiencing worst case
env.v0 = 0.4  # source exit velocity
env.a0 = 0.1
env.entrainment = [0.15, 0.2]
temperature = []
salinity = []
height = []

for t in T:
    env.solve(t=t)

    temp = np.round(env.theta(t=t), 2)  # only grab up to highest peak
    idx = np.asarray([i for i in range(len(s))])[temp < 0.01]
    try:
        idx = idx[0]
    except IndexError:
        idx = -1

    # Truncate the model up to the highest peak
    env._model[t] = env._model[t][:idx, :]

    tem = env.f(t=t) / (env.q(t=t) * 9.81 * 10**(-4)) + \
        tprof(env.z_disp(t=t))
    rho = eos_rho(tprof(env.z_disp(t=t)), sprof(env.z_disp(t=t)))
    tr = 2.13 * 10**(-4) * (tem - 2)
    salt = (rho - 1.041548 + tr) / (7.5 * 10**(-4)) + 34.89

    height += [h for h in env.z_disp(t=t)]
    temperature += [tm for tm in tem]
    salinity += [sal for sal in salt]

    # plot plume envelopes
    plt.plot(env.x_disp(t=t), env.z_disp(t=t))
    le, _, re = env.envelope(t=t)
    verts = [*zip(le[0], le[1]), *zip(reversed(re[0]), reversed(re[1]))]
    poly = Polygon(verts, facecolor='b', edgecolor='b', alpha=0.5)
    plt.gca().add_patch(poly)

    plt.plot(env.x_disp(t=t) + 150, env.z_disp(t=t))
    verts = [*zip(le[0] + 150, le[1]), *zip(reversed(re[0] + 150), reversed(re[1]))]
    poly = Polygon(verts, facecolor='orange', edgecolor='orange', alpha=0.5)
    plt.gca().add_patch(poly)

    plt.title("Plume Envelopes, U=" + str(np.round(curfunc(0, t), 4)))
    plt.xlabel("X in crossflow direction (meters)")
    plt.ylabel("Z (meters)")
    plt.xlim([-50, 500])
    plt.ylim([0, 180])
    plt.show()

# observe for some time, the intrusion layer anomaly
plt.scatter(height, temperature - tprof(np.asarray(height)))
plt.xlabel("Water column height")
plt.ylabel("Temperature anomaly")
plt.show()

plt.scatter(height, salinity - sprof(np.asarray(height)))
plt.xlabel("Water column height")
plt.ylabel("Salinity anomaly")
plt.show()
