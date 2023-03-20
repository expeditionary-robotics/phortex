"""YBW Post-Occupy Line Sentry Dive"""

"""
We have a budget of 2.25 hrs guaranteed for on-station pattern; and up to 3 hrs possible. 
We will plan a 5 hour mission, front-loading the most important "stuff" within the time window.
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
REFERENCE = (9.8, -104.31666667, 2555)  # Sentry origin
FLIGHT_ORIGIN = (9.838117, -104.291332)  # Where we want the flight to center
MIN_DEPTH = -2515  # bio9 2512, pvent 2510, lip 2504
VEL = 1.0
SAFETY = 0.5  # hours of additional buffer to add to time estimate
BATHY = get_bathy(lat_min=9.837156,
                  lat_max=9.83922,
                  lon_min=-104.292402,
                  lon_max=-104.290257,
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

lawn1 = Lawnmower(t0=0,
                  vel=VEL,
                  lh=180,
                  lw=180,
                  resolution=8,
                  altitude=20,
                  origin=(FX+90, FY+90),
                  orientation=180,
                  noise=None)
print(lawn1.length)
lawn2 = Lawnmower(t0=0,
                  vel=VEL,
                  lh=180,
                  lw=180,
                  resolution=12,
                  altitude=20,
                  origin=(FX-90, FY+90),
                  orientation=-90,
                  noise=None)
# lawn2 = Lawnmower(t0=0,
#                   vel=VEL,
#                   lh=200,
#                   lw=200,
#                   resolution=15,
#                   altitude=40,
#                   origin=(FX+100, FY-100),
#                   orientation=90,
#                   noise=None)
print(lawn2.length)
transit = np.sqrt((lawn1.xcoords[-1] - lawn2.xcoords[0])**2 + (lawn1.ycoords[-1] - lawn2.ycoords[0])**2)
print(transit)
dive1_mission = Chain(t0=0.,
                      vel=VEL,
                      traj_list=[lawn1, lawn2],
                      altitude=[20, 20],
                      noise=None)
# Convert to lat-lon points
DLAT = []
DLON = []
for coordx, coordy in zip(dive1_mission.path.xy[0], dive1_mission.path.xy[1]):
    dlat, dlon = utm.to_latlon(coordx, coordy, RN, RL)
    DLAT.append(dlat)
    DLON.append(dlon)
altitude = dive1_mission.zcoords
m = [a + MIN_DEPTH for a in dive1_mission.zcoords]
print(m)

dive_plot = go.Scatter3d(x=DLON,
                         y=DLAT,
                         z=m,
                         mode="lines",
                         name="Dive")
vent_plot = go.Scatter3d(x=[-104.2912886, -104.2912481],
                         y=[9.8384848, 9.8379783],
                         z=[-2510, -2512],
                         mode="markers",
                         name="Vents")
fig = go.Figure([bathy_plot, dive_plot, vent_plot])
fig.update_layout(scene_camera=camera, showlegend=False)
fig.update_yaxes(scaleanchor="x", scaleratio=1)
fig.show()

print(((lawn1.length + lawn2.length + transit) / VEL / 3600.) + SAFETY)
# print(dive1_mission.length / VEL / 3600.)

np.savetxt(f"./fumes/examples/d664_wpts_{np.mean(m)}.txt",
           np.asarray([DLAT, DLON]).T,
           delimiter=" ",
           fmt=["%4.6f", "%4.6f"])
