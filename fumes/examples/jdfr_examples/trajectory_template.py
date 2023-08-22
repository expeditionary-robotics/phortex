"""Template code for planning lawnmower or spiral patterns.

Yields a waypoint format that the Sentry team can ingest.

Usage:
- Modify top matter
- run `python trajectory_template.py` in the terminal.

Created by: Victoria Preston
Last Modified: August 2023
"""

import matplotlib.pyplot as plt
import utm
import numpy as np
from fumes.trajectory import Lawnmower, Spiral
from fumes.trajectory import Chain
from fumes.environment import Extent
from fumes.environment.utils import get_bathy
import plotly.graph_objs as go


# Key geographic information
REFERENCE = (9.8, -104.31666667, 2555)  # Sentry origin, get from Sentry team
FLIGHT_ORIGIN = (9.9055599, -104.294361)  # Where we want to scenter the map
MIN_DEPTH = -2551.37
VEL = 1.0  # in m/s -- remember that a knot is 0.5m/s
BATHY = get_bathy(lat_min=9.904768,
                  lat_max=9.906458,
                  lon_min=-104.294727,
                  lon_max=-104.293951,
                  rsamp=0.1, buffer=0.001)
bathy_plot = go.Mesh3d(x=BATHY.lon, y=BATHY.lat, z=BATHY.depth,
                       intensity=BATHY.depth,
                       colorscale='Viridis',
                       opacity=1.0,
                       name="Bathy")
name = 'eye = (x:0., y:0., z:2.5)'
camera = dict(
    eye=dict(x=1.25, y=1.25, z=0.1),
    center=dict(x=0, y=0, z=-0.2)
)

# Convert to xy coordinates for planning
RX, RY, RN, RL = utm.from_latlon(REFERENCE[0], REFERENCE[1])
FX, FY, FN, FL = utm.from_latlon(FLIGHT_ORIGIN[0], FLIGHT_ORIGIN[1])


spiral1 = Spiral(t0=0,
                 vel=VEL,
                 lh=160,
                 lw=100,
                 resolution=5,
                 altitude=20,
                 origin=(FX, FY),
                 orientation=0,
                 noise=None)
spiral1.reverse()

spiral2 = Spiral(t0=0,
                 vel=VEL,
                 lh=200,
                 lw=140,
                 resolution=5,
                 altitude=40,
                 origin=(FX, FY),
                 orientation=0,
                 noise=None)

dive1_mission = Chain(t0=0.,
                      vel=VEL,
                      traj_list=[spiral1, spiral2],
                      altitude=[20, 40],
                      noise=None)

# Convert to lat-lon points
DLAT = []
DLON = []
for coordx, coordy in zip(dive1_mission.path.xy[0], dive1_mission.path.xy[1]):
    dlat, dlon = utm.to_latlon(coordx, coordy, RN, RL)
    DLAT.append(dlat)
    DLON.append(dlon)
altitude = dive1_mission.zcoords

dive_plot = go.Scatter3d(x=DLON,
                         y=DLAT,
                         z=[a + MIN_DEPTH for a in dive1_mission.zcoords],
                         mode="lines",
                         name="Dive")
fig = go.Figure([bathy_plot, dive_plot])
fig.update_layout(scene_camera=camera, showlegend=False)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()

print(dive1_mission.length / VEL / 3600.)

#################

lawn1 = Lawnmower(t0=0,
                  vel=VEL,
                  lh=180,
                  lw=180,
                  resolution=5,
                  altitude=20,
                  origin=(FX-90, FY-90),
                  orientation=0,
                  noise=None)
print(lawn1.length)
lawn2 = Lawnmower(t0=0,
                  vel=VEL,
                  lh=180,
                  lw=180,
                  resolution=5,
                  altitude=20,
                  origin=(FX-90, FY+90),
                  orientation=-90,
                  noise=None)
print(lawn2.length)
print(np.sqrt((lawn1.xcoords[-1] - lawn2.xcoords[0])**2 + (lawn1.ycoords[-1] - lawn2.ycoords[0])**2))
lawn3 = Lawnmower(t0=0,
                  vel=VEL,
                  lh=200,
                  lw=160,
                  resolution=5,
                  altitude=40,
                  origin=(FX+100, FY-80),
                  orientation=90,
                  noise=None)
lawn4 = Lawnmower(t0=0,
                  vel=VEL,
                  lh=160,
                  lw=200,
                  resolution=5,
                  altitude=40,
                  origin=(FX-100, FY-80),
                  orientation=0,
                  noise=None)
print(lawn3.length)
print(lawn4.length)
print(np.sqrt((lawn3.xcoords[-1] - lawn4.xcoords[0])**2 + (lawn3.ycoords[-1] - lawn4.ycoords[0])**2))
dive1_mission = Chain(t0=0.,
                      vel=VEL,
                      traj_list=[lawn1, lawn2, lawn3, lawn4],
                      altitude=[20, 20, 40, 40],
                      noise=None)

# Convert to lat-lon points
DLAT = []
DLON = []
for coordx, coordy in zip(dive1_mission.path.xy[0], dive1_mission.path.xy[1]):
    dlat, dlon = utm.to_latlon(coordx, coordy, RN, RL)
    DLAT.append(np.round(dlat, 6))
    DLON.append(np.round(dlon, 6))
altitude = dive1_mission.zcoords
depth_coords = [np.round(a + MIN_DEPTH, 2) for a in altitude]

dive_plot = go.Scatter3d(x=DLON,
                         y=DLAT,
                         z=[a + MIN_DEPTH for a in dive1_mission.zcoords],
                         mode="lines",
                         name="Dive")
fig = go.Figure([bathy_plot, dive_plot])
fig.update_layout(scene_camera=camera, showlegend=True)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()

print(dive1_mission.length / VEL / 3600.)

# Save the lat-lon coordinates with depth saved
np.savetxt("./fumes/examples/epr_examplesybw_plume_dive.txt",
           np.asarray([DLAT, DLON]).T,
           delimiter=" ",
           fmt=["%4.6f", "%4.6f"])
