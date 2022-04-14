"""Utility functions for simulation and plotting."""
import os
import yaml
import utm
import numpy as np
import pandas as pd
import pickle

# Import dependencies
import plotly
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from fumes.utils import convert_to_latlon

import pdb

# Standard color sequence
COLORS = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
          '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']


def translation(deltax, deltay, deltaz):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [deltax, deltay, deltaz, 1],
    ])


def scaling(cx, cy, cz):
    return np.array([
        [cx, 0, 0, 0],
        [0, cy, 0, 0],
        [0, 0, cz, 0],
        [0, 0, 0, 1],
    ])


def yrotation(theta):
    return np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 0, 1],
    ])


def xrotation(theta):
    return np.array([
        [1, 0, 0, 0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta), np.cos(theta), 0],
        [0, 0, 0, 1],
    ])


def zrotation(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta), np.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def scatter_obj(data, name, color="blue", opacity=1.0, size=5):
    return {
        "x": data[:, 0],
        "y": data[:, 1],
        "z": data[:, 2],
        "mode": "markers",
        "marker": {
            'size': size,
            'opacity': opacity,
            'color': color
        },
        "name": name,
        "type": "scatter3d"
    }


def mesh_obj(data, name, return_obj=False):
    if return_obj:
        return go.Mesh3d(
            x=data[:, 0].flatten(),
            y=data[:, 1].flatten(),
            z=data[:, 2].flatten(),
            intensity=data[:, 2].flatten(),
            colorscale="Viridis",
            colorbar=dict(thickness=30, x=-0.1),
            opacity=0.5,
            name=name,
        )
    return {
        "x": data[:, 0].flatten(),
        "y": data[:, 1].flatten(),
        "z": data[:, 2].flatten(),
        "intensity": data[:, 2].flatten(),
        "colorscale": "Viridis",
        "opacity": 0.3,
        "name": name,
        "type": "mesh3d"
    }


def draw_ellipse(a, b, n=200):
    # Sample points on a circle in the xy plane
    rad = np.linspace(0, 2 * np.pi, n)
    x = b * np.cos(rad)
    y = a * np.sin(rad)
    return x, y


def to_homogenous(x, y):
    return np.vstack([x, y, np.zeros(y.shape), np.ones(y.shape)]).T


def frame_args(duration):
    return {"frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"}}


def _make_and_configure_fig(title, xlim, ylim, zlim):
    # Make a figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": [],
        "type": "scatter3d"
    }
    # Layout
    fig_dict["layout"]["title"] = title
    fig_dict["layout"]["width"] = 1500
    fig_dict["layout"]["height"] = 1500
    fig_dict["layout"]["scene"] = dict(
        xaxis=dict(nticks=10, range=xlim,),
        yaxis=dict(nticks=10, range=ylim,),
        zaxis=dict(nticks=10, range=zlim,),)

    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, frame_args(50)],
                    "label": "&#9654;",  # play symbol
                    "method": "animate",
                },
                {
                    "args": [[None], frame_args(0)],
                    "label": "&#9724;",  # pause symbol
                    "method": "animate",
                },
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "type": "buttons",
                    "x": 0.1,
                    "y": 0,
        }]

    return fig_dict


def _get_sliders(fig_dict):
    # Configure the sliders
    return [{
        "pad": {"b": 10, "t": 60},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [
            {
                "args": [[f["name"]], frame_args(0)],
                "label": str(k),
                "method": "animate",
            }
            for k, f in enumerate(fig_dict["frames"])
        ],
    }]


def plot_window_in_bathy(env, bathy, sites, xlim, ylim, zlim):
    """Plot an x, y, z extent cuboid in the global bathymap."""
    # Convert extent to latitude and longitude
    lim = np.vstack([[xlim[0], ylim[0], zlim[0]], [xlim[1], ylim[1], zlim[1]]])
    lim = convert_to_latlon(lim, env.extent.origin)
    xlim = lim[:, 0]
    ylim = lim[:, 1]
    zlim = lim[:, 2]

    points = np.array([
        [xlim[0], ylim[0], zlim[0]],
        [xlim[0], ylim[1], zlim[0]],
        [xlim[1], ylim[1], zlim[0]],
        [xlim[1], ylim[0], zlim[0]],
        [xlim[0], ylim[0], zlim[1]],
        [xlim[0], ylim[1], zlim[1]],
        [xlim[1], ylim[1], zlim[1]],
        [xlim[1], ylim[0], zlim[1]],
    ])
    bounds_data = go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                               mode="markers")
    cuboid_data = go.Mesh3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
                            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
                            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
                            color='#DC143C',
                            opacity=0.6, flatshading=True)

    x = bathy['long'].values
    y = bathy['lat'].values
    z = bathy['depth'].values
    bathy_data = go.Mesh3d(x=x[::100], y=y[::100], z=-z[::100],
                           intensity=-z[::100], colorscale='Viridis',
                           opacity=0.50)

    sites_data = go.Scatter3d(x=sites[:, 0], y=sites[:, 1], z=sites[:, 2],
                              mode="markers",
                              marker=dict(
                                   size=5,
        color="orange",
        opacity=0.9,)
    )

    fig = go.Figure(data=[bathy_data, cuboid_data, sites_data])
    return fig


def scatter_plume_and_traj(times, env, coord_times, coords, bathy=None, sites=None,
                           title="Plume Evolution", xlim=[-900, 900],
                           ylim=[-900, 900], zlim=[0, 100], ref_global=False):
    """Generate 3D scatter plot of plume and trajectory."""
    if bathy is not None and not ref_global:
        raise ValueError("Bathy can only be plotted in the global reference.")
    if sites is not None and not ref_global:
        raise ValueError("Sites can only be plotted in the global reference.")

    if ref_global:
        # Convert extent to latitude and longitude
        lim = np.vstack([[xlim[0], ylim[0], zlim[0]], [xlim[1], ylim[1], zlim[1]]])
        lim = convert_to_latlon(lim, env.extent.origin)
        xlim = lim[:, 0]
        ylim = lim[:, 1]
        zlim = lim[:, 2]

    # zlim = [-2100, -1000]
    # Generate figure skeleton
    fig_dict = _make_and_configure_fig(
        title, xlim, ylim, zlim)

    # Initial simulation frame; final plume time
    pts_plume = env.get_pointcloud(t=times[-1])
    if ref_global:
        pts_plume = convert_to_latlon(pts_plume, env.extent.origin)
    pts_traj = coords[coord_times < times[-1]]

    # Process bathy data
    if ref_global:
        # Window the bathy
        bathy = bathy[(bathy['long'] >= xlim[0]) & (bathy['long'] <= xlim[1])]
        bathy = bathy[(bathy['lat'] >= ylim[0]) & (bathy['lat'] <= ylim[1])]
        pts_bathy = bathy[['long', 'lat', 'depth']].values

    # Create 3D scatter objects
    if ref_global:
        data3 = mesh_obj(pts_bathy, name="Bathy")
        data4 = scatter_obj(sites, name="Sites", color="orange", size=5)
        fig_dict["data"].append(data3)
        fig_dict["data"].append(data4)

    data1 = scatter_obj(pts_plume, name="Plume", color="blue")
    data2 = scatter_obj(pts_traj, name="Trajectory", color="red", opacity=1.0)
    fig_dict["data"].append(data1)
    fig_dict["data"].append(data2)

    # Fill in each frame of the simulation
    # Start from time 1, since time 0 will have no trajectory by definition
    for i, t in enumerate(times[1:]):
        # Initialize frame
        frame = {"data": [], "name": f"t{t}"}

        # Get plume point cloud
        pts_plume = env.get_pointcloud(t=t)
        if ref_global:
            pts_plume = convert_to_latlon(pts_plume, env.extent.origin)
        pts_traj = coords[coord_times < t]

        # Create 3D scatter objects
        if ref_global:
            data3 = mesh_obj(pts_bathy, name="Bathy")
            data4 = scatter_obj(sites, name="Sites", color="orange", size=5)
            frame["data"].append(data3)
            frame["data"].append(data4)
        data1 = scatter_obj(pts_plume, name="Plume", color="blue")
        data2 = scatter_obj(pts_traj, name="Trajectory", color="red", opacity=1.0)
        frame["data"].append(data1)
        frame["data"].append(data2)

        # Add to frame list
        fig_dict["frames"].append(frame)

    fig_dict["layout"]["sliders"] = _get_sliders(fig_dict)

    # Create plotly figure
    fig = go.Figure(fig_dict)
    return fig


def plot_trajectory_map(trajectory, extent, site="ridge"):
    """Plots bathy, sites, and other relevant information with trajectory object."""
    bathy = get_bathy(extent)
    bathy_fig = plot_bathy_underlay(extent, site=site)
    sites_fig = plot_sites_overlay(extent)

    coords = np.asarray(trajectory.uniformly_sample(0.5))
    easting, northing, z1, z2 = utm.from_latlon(extent.origin[0], extent.origin[1])
    xcoords = coords.T[1, :] + easting
    ycoords = coords.T[2, :] + northing
    print(xcoords)
    try:
        lat, lon = utm.to_latlon(xcoords, ycoords, z1, z2)
    except:
        lat, lon = utm.to_latlon(np.float32(xcoords), np.float32(ycoords), z1, z2)
    traj_fig = go.Scatter(x=lon, y=lat)
    # knob_data = go.Scatter(y=[27.41364], x=[-111.3758], mode="markers")

    fig = go.Figure(data=[bathy_fig, sites_fig, traj_fig], layout=dict(width=900, height=900))
    fig.show()


def get_bathy(extent, site='ridge', latlon_provided=False, step=100):
    """Get and window the bathy file."""
    if site == 'ridge':
        bathy_file = os.path.join(os.getenv("SENTRY_DATA"), f"processed/ridge.txt")
        bathy = pd.read_table(bathy_file, names=["long", "lat", "depth"]).dropna()
    elif site == 'ring':
        bathy_file = os.path.join(os.getenv("SENTRY_DATA"), f"processed/ring.txt")
        bathy = pd.read_table(bathy_file, names=["long", "lat", "depth"]).dropna()
    elif site == 'plain':
        bathy_file = os.path.join(os.getenv("SENTRY_DATA"), f"processed/plain.txt")
        bathy = pd.read_table(bathy_file, names=["long", "lat", "depth"]).dropna()
    elif site == 'all':
        ridge_file = os.path.join(os.getenv("SENTRY_DATA"), f"processed/ridge.txt")
        ring_file = os.path.join(os.getenv("SENTRY_DATA"), f"processed/ring.txt")
        ridge_bathy = pd.read_table(ridge_file, names=["long", "lat", "depth"]).dropna()
        ring_bathy = pd.read_table(ring_file, names=["long", "lat", "depth"]).dropna()
        bathy = pd.concat([ridge_bathy, ring_bathy])
    else:
        return None
    
    # bathy.set_index("Unnamed: 0", inplace=True)

    if latlon_provided is False:
        xlim, ylim, zlim = extent_to_lat_lon(extent)
    else:
        xlim = extent.xrange
        ylim = extent.yrange
    bathy = bathy[(bathy['long'] >= xlim[0]) & (bathy['long'] <= xlim[1])]
    bathy = bathy[(bathy['lat'] >= ylim[0]) & (bathy['lat'] <= ylim[1])]
    bathy = bathy[::step]
    return bathy


def plot_bathy_underlay(extent, site="ridge", step=10000, latlon_provided=False):
    """Provides a GO figure for bathy underlaying."""
    bathy = get_bathy(extent, site=site, latlon_provided=latlon_provided)
    x = bathy['long'].values
    y = bathy['lat'].values
    z = bathy['depth'].values
    return go.Contour(x=x[::step], y=y[::step], z=z[::step], colorscale='Viridis', ncontours=10)


def get_sites(extent, bathy, latlon_provided=False):
    if latlon_provided is False:
        xrange, yrange, zrange = extent_to_lat_lon(extent)
    else:
        xrange = extent.xrange
        yrange = extent.yrange
    sites_file = os.path.join(os.getenv("SENTRY_DATA"), f"site_data.yaml")
    sites = yaml.load(open(sites_file), Loader=yaml.FullLoader)

    site_locs = []
    for x in sites.values():
        if x['lon'] <= xrange[1] and x['lon'] >= xrange[0]:
            if x['lat'] <= yrange[1] and x['lat'] >= yrange[0]:
                site_locs.append((x['lon'], x['lat'], 0))
    site_locs = np.array(site_locs).reshape(-1, 3)

    # Find site depth from nearest bathy point
    for i in range(site_locs.shape[0]):
        idx = np.abs(bathy[['long', 'lat']] - site_locs[i, 0:2]).sum(axis=1).idxmin()
        d = bathy.loc[idx, "depth"]
        site_locs[i, 2] = -d
    return site_locs


def plot_sites_overlay(extent, latlon_provided=False):
    """Provides a GO figure for sites overlay."""
    xrange, yrange, zrange = extent_to_lat_lon(extent)
    sites_file = os.path.join(os.getenv("SENTRY_DATA"), f"site_data.yaml")
    sites = yaml.load(open(sites_file), Loader=yaml.FullLoader)
    site_locs = []
    for x in sites.values():
        if x['lon'] <= xrange[1] and x['lon'] >= xrange[0]:
            if x['lat'] <= yrange[1] and x['lat'] >= yrange[0]:
                site_locs.append((x['lon'], x['lat']))
    site_locs = np.array(site_locs).reshape(-1, 2)
    return go.Scatter(x=site_locs[:, 0], y=site_locs[:, 1], mode="markers")


def plot_sites_underlay(extent, latlon_provided=False):
    """Provides a GO figures for sites underlay."""
    site_locs = get_sites(extent, latlon_provided=latlon_provided)
    return go.Scatter(x=site_locs[:, 0], y=site_locs[:, 1], mode="markers")


def extent_to_lat_lon(extent):
    # Convert extent to latitude and longitude
    lim = np.vstack([[extent.xrange[0], extent.yrange[0], extent.zrange[0]],
                     [extent.xrange[1], extent.yrange[1], extent.zrange[1]]])
    lim = convert_to_latlon(lim, extent.origin)
    xlim = lim[:, 0]
    ylim = lim[:, 1]
    zlim = lim[:, 2]
    return xlim, ylim, zlim


def plot_plume(times, model, title="Plume Dynamics"):
    """Generate 3D scatter plot of plume."""
    # Get plot limits in lat-lon coordinates
    xlim, ylim, zlim = extent_to_lat_lon(model.extent)

    # Generate figure skeleton
    fig_dict = _make_and_configure_fig(
        title, xlim, ylim, zlim)

    # Initial simulation frame; final plume time
    # Draw a vertical slice of the plume envelope
    # model.solve(t=times[-1], overwrite=True)
    # z = model.odesys.z_disp(times[-1])
    # le, cl, re = model.odesys.envelope(t=times[-1])
    # plt.plot(*cl, label="Centerline")
    # plt.plot(*le, label="Left Extent")
    # plt.plot(*re, label="Right Extent")
    # plt.title("Plume Envelope")
    # plt.xlabel("X (meters)")
    # plt.ylabel("Z (meters)")
    # plt.legend()
    # plt.show()

    pts_plume = model.odesys.get_pointcloud(t=times[-1])
    pts_plume = convert_to_latlon(pts_plume, model.extent.origin)

    # Get the bathy data
    bathy = get_bathy(model.extent)
    pts_bathy = bathy[['long', 'lat', 'depth']].values

    # Get the sites data
    sites = get_sites(model.extent, bathy)

    # Create 3D scatter objects
    data1 = mesh_obj(pts_bathy, name="Bathy")
    data2 = scatter_obj(sites, name="Sites", color="orange", size=5)
    fig_dict["data"].append(data1)
    fig_dict["data"].append(data2)

    data3 = scatter_obj(pts_plume, name="Plume", color="blue")
    fig_dict["data"].append(data3)

    # Fill in each frame of the simulation
    for i, t in enumerate(times):
        print(f"Plotting time {t}, {i} of {len(times)}")
        # Initialize frame
        frame = {"data": [], "name": f"t{t}"}

        # Get plume point cloud
        pts_plume = model.odesys.get_pointcloud(t=t)
        pts_plume = convert_to_latlon(pts_plume, model.extent.origin)

        # Create 3D scatter objects
        # data1 = mesh_obj(pts_bathy, name="Bathy")
        # data2 = scatter_obj(sites, name="Sites", color="orange", size=5)
        data3 = scatter_obj(pts_plume, name="Plume", color="blue")
        frame["data"].append(data1)
        frame["data"].append(data2)
        frame["data"].append(data3)

        # Add to frame list
        fig_dict["frames"].append(frame)

    fig_dict["layout"]["sliders"] = _get_sliders(fig_dict)

    # Create plotly figure
    fig = go.Figure(fig_dict)
    return fig


def plot_centerline(times, model, title="Plume Dynamics"):
    """Generate time series of plume centerline."""
    for t in times:
        model.solve(t=t, overwrite=True)
        z = model.odesys.z_disp(t)
        le, cl, re = model.odesys.envelope(t=t)
        plt.plot(*cl, label=f"t={t}")
    plt.title("Plume Evolution")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    plt.show()

def visualize_and_save_traj(trajectory, extent, traj_name="temp_traj"):
    """Visualize trajectory in global cooordinates and save path to file.

    Args:
        traj (Trajectory): input Trajectory object
        traj_name (str):  name of the output trajectory object
        extent (Extent): an Extent object for mapping
    """
    REFERENCE = (float(os.getenv("LAT")),
                float(os.getenv("LON")),
                float(os.getenv("DEP")))

    EAST_REFERENCE, NORTH_REFERENCE, ZONE_NUM, ZONE_LETT = utm.from_latlon(
        REFERENCE[0], REFERENCE[1])

    with open(traj_name, "wb") as fh:
        pickle.dump(trajectory, fh)

    plot_trajectory_map(trajectory, extent)

    # Convert trajectory to sentry file
    print("Converting trajectory to Sentry mission file...")
    path_x = trajectory.path.xy[0] + EAST_REFERENCE
    path_y = trajectory.path.xy[1] + NORTH_REFERENCE

    # convert to lat lon
    map_lat, map_lon = utm.to_latlon(path_x, path_y, ZONE_NUM, ZONE_LETT)
    save_latlon = np.around(np.asarray([map_lat, map_lon]).T, decimals=5)

    # save file with name of depth
    np.savetxt(os.path.join(os.getenv("SENTRY_DATA"), traj_name),
            save_latlon, delimiter=' ', fmt='%1.5f')

if __name__ == "__main__":
    from fumes.environment.extent import Extent
    REFERENCE = (float(os.getenv("LAT")),
                float(os.getenv("LON")),
                float(os.getenv("DEP")))
    extent = Extent(xrange=(0, 2500),
                xres=50,
                yrange=(500, 2500),
                yres=50,
                zrange=(0, 300),
                zres=100,
                global_origin=REFERENCE)
    bathy = get_bathy(extent)
    x = bathy['long'].values
    y = bathy['lat'].values
    z = bathy['depth'].values
    bathy_data = go.Mesh3d(x=x, y=y, z=z,
                            intensity=z,
                            colorscale='Viridis',
                            opacity=0.50,
                            name="Bathy")
    b = go.Figure(data=[bathy_data], layout_title_text="Bathymetry for Site Ridge")
    b.show()