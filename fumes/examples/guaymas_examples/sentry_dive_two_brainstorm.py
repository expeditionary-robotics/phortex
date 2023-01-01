"""Creates a list of ship multibeam paths."""
import matplotlib.pyplot as plt
import numpy as np
import utm
import yaml
import os
import pandas as pd
import plotly.graph_objs as go
from fumes.trajectory.lawnmower import Lawnmower, Spiral
from fumes.trajectory import Chain
from fumes.environment.extent import Extent
from fumes.simulator.utils import plot_bathy_underlay, plot_sites_overlay, plot_trajectory_map

def save_mission_to_file(traj, alt, name, EAST_REFERENCE, NORTH_REFERENCE, ZONE_NUM, ZONE_LETT):
    # Convert trajectory to sentry file
    print("Converting trajectory to Sentry mission file...")
    path_x = traj.path.xy[0] + EAST_REFERENCE
    path_y = traj.path.xy[1] + NORTH_REFERENCE

    # convert to lat lon
    map_lat, map_lon = utm.to_latlon(path_x, path_y, ZONE_NUM, ZONE_LETT)
    save_latlon = np.around(np.asarray([map_lat, map_lon]).T, decimals=5)

    # save file with name of depth
    traj_name = f"{name}" + str(alt) + "m_planned.txt"
    np.savetxt(os.path.join(os.getenv("SENTRY_DATA"), traj_name),
            save_latlon, delimiter=' ', fmt='%4.5f')

    print("Planning complete.")

if __name__ == '__main__':
    reference = (27.4, -111.4, 1800)
    reference_easting, reference_northing, zn, zl = utm.from_latlon(reference[0], reference[1])
    lowres_flyover_lawn = Lawnmower(t0=0.,
                                    vel=0.75,
                                    lh=1500,
                                    lw=1600,
                                    resolution=100,
                                    altitude=0.,
                                    origin=(600, 700),
                                    orientation=-20.,
                                    noise=None)
    extent = Extent(xrange=(0, 3500),
                    xres=100,
                    yrange=(0, 2500),
                    yres=100,
                    zrange=(0, 1800),
                    zres=10,
                    global_origin=reference)
    plot_trajectory_map(lowres_flyover_lawn, extent)
    print("Trajectory length: ", lowres_flyover_lawn.length)
    print("Trajectory time: ", lowres_flyover_lawn.length / lowres_flyover_lawn.vel / 3600.)
    save_mission_to_file(lowres_flyover_lawn, 120.0, "lowres_plain_sweep", reference_easting, reference_northing, zn, zl)

    # print(lowres_flyover_lawn.coords[-1])
    # import pdb; pdb.set_trace()

    lowres_flyover_ridge = Lawnmower(t0=0.,
                                     vel=0.75,
                                     lh=600,
                                     lw=1600,
                                     resolution=100,
                                     altitude=0.,
                                     origin=(lowres_flyover_lawn.path.xy[0][-1], lowres_flyover_lawn.path.xy[1][-1]),
                                     orientation=-110.,
                                     noise=None)
    extent = Extent(xrange=(0, 3500),
                    xres=100,
                    yrange=(0, 2500),
                    yres=100,
                    zrange=(0, 1800),
                    zres=10,
                    global_origin=reference)
    plot_trajectory_map(lowres_flyover_ridge, extent)
    print("Trajectory length: ", lowres_flyover_ridge.length)
    print("Trajectory time: ", lowres_flyover_ridge.length / lowres_flyover_ridge.vel / 3600.)
    save_mission_to_file(lowres_flyover_ridge, 45.0, "lowres_ridge_sweep", reference_easting, reference_northing, zn, zl)

    chimney2 = (27.41236, -111.3861)
    chim_east, chim_north, _, _ = utm.from_latlon(chimney2[0], chimney2[1])
    highres_flyover_chim = Spiral(t0=0.,
                                  vel=0.75,
                                  lh=200,
                                  lw=200,
                                  resolution=30,
                                  altitude=0.,
                                  origin=(chim_east - reference_easting,
                                          chim_north - reference_northing),
                                  orientation=-20.,
                                  noise=None)
    plot_trajectory_map(highres_flyover_chim, extent)
    print("Trajectory length: ", highres_flyover_chim.length)
    print("Trajectory time: ", highres_flyover_chim.length / highres_flyover_chim.vel / 3600.)
    save_mission_to_file(highres_flyover_chim, 100.0, "highres_chimney_spiral", reference_easting, reference_northing, zn, zl)

    knob = (27.41364, -111.3758)
    knob_east, knob_north, _, _ = utm.from_latlon(knob[0], knob[1])
    highres_flyover_knob = Spiral(t0=0.,
                                  vel=0.75,
                                  lh=200,
                                  lw=200,
                                  resolution=30,
                                  altitude=0.,
                                  origin=(knob_east - reference_easting,
                                          knob_north - reference_northing),
                                  orientation=-20.,
                                  noise=None)
    plot_trajectory_map(highres_flyover_knob, extent)
    print("Trajectory length: ", highres_flyover_knob.length)
    print("Trajectory time: ", highres_flyover_knob.length / highres_flyover_knob.vel / 3600.)
    save_mission_to_file(highres_flyover_knob, 30.0, "highres_knob_spiral", reference_easting, reference_northing, zn, zl)

    chain = Chain(t0=0, vel=0.75, traj_list=[lowres_flyover_lawn, lowres_flyover_ridge, highres_flyover_chim, highres_flyover_knob])

    print("Trajectory length: ", chain.length)
    print("Trajectory time: ", chain.length / chain.vel / 3600.)
    plot_trajectory_map(chain, extent)

    # bathy_data = plot_bathy_underlay(extent, latlon_provided=True)
    # site_data = plot_sites_overlay(extent, latlon_provided=True)
    # # lawn_data = go.Scatter(x=lon, y=lat)
    # lawn_data = go.Scatter(y=[27.40844, 27.40976, 27.41108], x=[-111.3892, -111.382, -111.376], mode="markers")
    # knob_data = go.Scatter(y=[27.41364], x=[-111.3758], mode="markers")
    # fig = go.Figure(data=[bathy_data, lawn_data, site_data, knob_data], layout=dict(width=900, height=900))
    # fig.show()
