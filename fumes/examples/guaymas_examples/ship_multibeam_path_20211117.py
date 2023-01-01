"""Creates a list of ship multibeam paths."""
import matplotlib.pyplot as plt
import numpy as np
import utm
import yaml
import os
import pandas as pd
import plotly.graph_objs as go
from fumes.trajectory.lawnmower import Lawnmower
from fumes.environment.extent import Extent
from fumes.simulator.utils import plot_bathy_underlay, plot_sites_overlay, plot_trajectory_map

if __name__ == '__main__':
    chimney1 = (27.4, -111.4, 1800)
    easting, northing, z1, z2 = utm.from_latlon(chimney1[0], chimney1[1])
    lawn = Lawnmower(t0=0.,
                     vel=1000 / 60. / 60.,
                     lh=1000,
                     lw=100,
                     resolution=100,
                     altitude=0.,
                     origin=(500, 500),
                     orientation=0.,
                     noise=None)
    coords = np.asarray(lawn.uniformly_sample(100))
    xcoords = coords.T[1, :] + easting
    ycoords = coords.T[2, :] + northing
    lat, lon = utm.to_latlon(xcoords, ycoords, z1, z2)
    padding = 0.015
    extent = Extent(xrange=(0, 5000),
                    xres=100,
                    yrange=(0, 5000),
                    yres=100,
                    zrange=(0, 1800),
                    zres=10,
                    global_origin=chimney1)
    plot_trajectory_map(lawn, extent)
    # bathy_data = plot_bathy_underlay(extent, latlon_provided=True)
    # site_data = plot_sites_overlay(extent, latlon_provided=True)
    # # lawn_data = go.Scatter(x=lon, y=lat)
    # lawn_data = go.Scatter(y=[27.40844, 27.40976, 27.41108], x=[-111.3892, -111.382, -111.376], mode="markers")
    # knob_data = go.Scatter(y=[27.41364], x=[-111.3758], mode="markers")
    # fig = go.Figure(data=[bathy_data, lawn_data, site_data, knob_data], layout=dict(width=900, height=900))
    # fig.show()
