"""Creates a multiplume PHUMES simulation environment for YBW-Sentry."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import utm
import os

from fumes.environment.mtt import Multiplume, CrossflowMTT
from fumes.environment.utils import eos_rho
from fumes.environment.extent import Extent
from fumes.environment.utils import get_bathy
from fumes.environment.profile import Profile

from fumes.trajectory.lawnmower import Lawnmower
from fumes.trajectory.chain import Chain


def create_crossflow_world(locs, extent, s, tprof, sprof, rhoprof, curfunc, headfunc, v0s, a0s, s0s, t0s, rho0s, lam, entrainment):
    """Creates a list of crossflow worlds."""
    envs = []
    for loc, v0, a0, s0, t0, rho0 in zip(locs, v0s, a0s, s0s, t0s, rho0s):
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
SMOKING_CHIMNEYS = [(9.905665, -104.294522, -2556.349),
                    (9.905230, -104.294360, -2550.3),
                    (9.905639, -104.294531, -2555.558)]
BATHY = get_bathy(lat_min=9.904768,
                  lat_max=9.906458,
                  lon_min=-104.294727,
                  lon_max=-104.293951,
                  rsamp=0.1, buffer=0.001)
Beast, Bnorth, _, _ = utm.from_latlon(BATHY.lat.values, BATHY.lon.values)
bathy_plot = go.Mesh3d(x=Beast, y=Bnorth, z=BATHY.depth,
                       intensity=BATHY.depth,
                       colorscale='Viridis',
                       opacity=1.0,
                       name="Bathy")
# name = 'eye = (x:0., y:0., z:2.5)'
# camera = dict(eye=dict(x=1.25, y=1.25, z=0.1),
#               center=dict(x=0, y=0, z=-0.2))

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
a0s = [0.25, 0.25, 0.25]

# source salinity; estimate
s0s = [34.608, 34.608, 34.608]

# source temperature; estimate
t0s = [270., 270., 270.]

# source density
rho0s = [eos_rho(t, s) for t, s, in zip(t0s, s0s)]  # source density

# assume they are all essentially symmetric
lam = 1.0  # for crossflow, major-minor axis ratio

# assume that the mixing coefficients are fixed for all plumes
entrainment = [0.12, 0.25]  # entrainment coeffs

# set the simulation boundaries
extent = Extent(xrange=(-100., 100.),
                xres=100,
                yrange=(-100., 100.),
                yres=100,
                zrange=(0, 100),
                zres=10,
                global_origin=(0, 0, 0))

# define the environments
envs = create_crossflow_world(locs=locs,
                              extent=extent,
                              s=s,
                              tprof=tprof,
                              sprof=sprof,
                              rhoprof=rhoprof,
                              curfunc=curfunc,
                              headfunc=headfunc,
                              v0s=v0s,
                              a0s=a0s,
                              s0s=s0s,
                              t0s=t0s,
                              rho0s=rho0s,
                              lam=lam,
                              entrainment=entrainment)
multiplume = Multiplume(envs)

# Get a plume intersection
y = np.linspace(-50, 50, 100)
x = np.zeros_like(y)
height = np.zeros(y.shape[0] * x.shape[0])
pq = multiplume.get_value(t=10, loc=(x, y, height), from_cache=False)
plt.plot(y, pq)
plt.xlabel('Y-coordinate')
plt.ylabel('Plume-State')
plt.title('Environment Slice at X=0')
plt.show()

# Get a birds-eye snapshot of plume probabilities
fig, ax = plt.subplots(len(query_times), 2, sharex=True, sharey=True)
if len(query_times) > 1:
    for i, qt in enumerate(query_times):
        ps = multiplume.get_snapshot(t=qt * 60. * 60., z=[20, 40])
        ax[i, 0].imshow(ps[0], origin="lower", extent=(-100, 100, -100, 100))
        ax[i, 0].set_ylabel('Y-coordinate')
        ax[i, 0].set_title("Z=20m, All Plumes")
        ax[i, 1].imshow(ps[1], origin="lower", extent=(-100, 100, -100, 100))
        ax[i, 1].set_ylabel('Y-coordinate')
        ax[i, 1].set_title("Z=40m, All Plumes")
else:
    ps = multiplume.get_snapshot(t=query_times[0] * 60. * 60., z=[20, 40])
    ax[0].imshow(ps[0], origin="lower", extent=(-100, 100, -100, 100))
    ax[0].set_ylabel('Y-coordinate')
    ax[0].set_title("Z=20m, All Plumes")
    ax[1].imshow(ps[1], origin="lower", extent=(-100, 100, -100, 100))
    ax[1].set_ylabel('Y-coordinate')
    ax[1].set_title("Z=40m, All Plumes")
plt.show()


# # Plot the 3D view of everything
env_3d_snapshot = multiplume.get_pointcloud(t=0.)
lat3d, lon3d = utm.to_latlon(env_3d_snapshot[:, 0] + RX, env_3d_snapshot[:, 1] + RY, RN, RL)
print(env_3d_snapshot)
env_fig = go.Scatter3d(x=env_3d_snapshot[:, 0] + RX,  # lon3d,
                       y=env_3d_snapshot[:, 1] + RY,  # lat3d,
                       z=REFERENCE[2] + env_3d_snapshot[:, 2],
                       mode="markers",
                       marker=dict(size=2, opacity=0.05, color='black'),
                       name=f"Environment Plume at t=0.")


x = np.linspace(np.nanmin(Beast), np.nanmax(Beast), 100)
y = np.linspace(np.nanmin(Bnorth), np.nanmax(Bnorth), 100)
X, Y = np.meshgrid(x, y)
# plane = go.Surface(x=X, y=Y, z=-2531 * np.ones_like(X), opacity=0.0, name="20m plane")
fig = go.Figure(data=[bathy_plot, env_fig])  # , layout=layout)
fig.update_layout(showlegend=True,
                  xaxis_title="Longitude",
                  yaxis_title="Latitude",
                  font=dict(size=18),
                  scene=dict(zaxis_title="",
                             xaxis_title="",
                             yaxis_title="",
                             aspectmode="data",
                             zaxis=dict(range=[-2575, -2450], tickfont=dict(size=20)),
                             yaxis=dict(tickfont=dict(size=20), autorange="reversed"),
                             xaxis=dict(tickfont=dict(size=20))),)
                #   scene_camera=camera)
fig.update_yaxes(scaleanchor="x", scaleratio=1.0)
fig.show()
