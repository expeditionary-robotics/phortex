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
from fumes.environment.utils import get_bathy_jdfr
import plotly.graph_objs as go

######
# Mission Params
######
# Key geographic information
REFERENCE = (49.0, -129.0, -2500)  # Sentry origin, get from Sentry team
FLIGHT_ORIGIN = (48.2, -129.4)  # Where we want to center the spiral
MIN_DEPTH = -2500
BATHY = get_bathy_jdfr("../jdfr-data-analysis/data/bathy/all_lowres.txt")
SUBSAMPLE = 50  # number of points in bathy to use for visualizing
bathy_plot = go.Mesh3d(x=BATHY.lon[::SUBSAMPLE], y=BATHY.lat[::SUBSAMPLE], z=BATHY.depth[::SUBSAMPLE],
                       intensity=BATHY.depth[::SUBSAMPLE],
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

# Sentry info
VEL = 1.0  # in m/s -- remember that a knot is 0.5m/s


########
# Main execution
########


if __name__ == '__main__':
    spiral1 = Spiral(t0=0,  # start "time"; does not matter for manual chaining
                     vel=VEL,  # sentry velocity, m/s
                     lh=2000,  # height of rectangle that describes the trajectory, m
                     lw=2000,  # width of rectangle that describes the trajectory, m
                     resolution=100,  # how tight the lines will be, m
                     altitude=100,  # height above the seafloor for execution, m
                     origin=(FX, FY),  # where the center of the spiral is
                     orientation=0,  # orientation of spiral
                     noise=None)  # whether to produce noise in the location
    spiral1.reverse()  # start at the tail, instead of the center

    spiral2 = Spiral(t0=0,
                     vel=VEL,
                     lh=5000,
                     lw=2000,
                     resolution=500,
                     altitude=150,
                     origin=(FX, FY),
                     orientation=0,
                     noise=None)
    
    lawn1 = Lawnmower(t0=0,
                      vel=VEL,
                      lh=5000,
                      lw=5000,
                      resolution=200,
                      altitude=110,
                      origin=(FX - 90, FY - 90),
                      orientation=0,
                      noise=None)

    dive_mission = Chain(t0=0.,
                         vel=VEL,
                         traj_list=[spiral1, spiral2, lawn1],
                         altitude=[100, 150, 110],  # pass the altitudes back in to the chain
                         noise=None)

    # Convert to lat-lon points
    DLAT = []
    DLON = []
    for coordx, coordy in zip(dive_mission.path.xy[0], dive_mission.path.xy[1]):
        dlat, dlon = utm.to_latlon(coordx, coordy, RN, RL)
        DLAT.append(dlat)
        DLON.append(dlon)
    altitude = dive_mission.zcoords
    depth_coords = [np.round(a + MIN_DEPTH, 2) for a in altitude]

    # Visualize the trajectory over the bathymetry
    dive_plot = go.Scatter3d(x=DLON,
                             y=DLAT,
                             z=[a + MIN_DEPTH for a in dive_mission.zcoords],
                             mode="lines",
                             name="Dive")
    fig = go.Figure([bathy_plot, dive_plot])
    fig.update_layout(scene_camera=camera, showlegend=True)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()

    # Print the duration, in hours, that the trajectory will take
    print(dive_mission.length / VEL / 3600.)

    # Save the lat-lon coordinates into a text file for Sentry team
    np.savetxt("./fumes/examples/jdfr_examples/test_trajectory.txt",
               np.asarray([DLAT, DLON]).T,
               delimiter=" ",
               fmt=["%4.6f", "%4.6f"])
