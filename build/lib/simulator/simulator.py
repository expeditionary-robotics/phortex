"""Simulator objects."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os
import copy
import yaml

from fumes.utils import data_home, output_home, tic, toc, convert_to_latlon
from .utils import scatter_plume_and_traj, plot_window_in_bathy


class Simulator(object):
    """Generic discrete-time simulation class."""

    def __init__(self, robot, environment, ref_global=False):
        """Initialize simulator.

        Args:
            robot (Robot): a Robot object
            environment (Environment): an Environment object
            ref_global (bool): if True, simulation plots are
                generated in a global reference frame.
        """
        self.rob = robot
        self.env = environment
        self.ref_global = ref_global  # whether to plot in global reference
        self.coords = None  # coordinates visited by robot
        self.obs = None  # observations taken by robot
        self.com_coords = None  # coordinates communicated online
        self.com_obs = None  # observations communicated online

    def simulate(self, times, experiment_name=None):
        """Perform simulation.

        Args:
            times (np.array): array of times to simulate
        """
        self.experiment_name = experiment_name
        self.times = times
        self.coords = np.zeros((len(times), 3))
        self.obs = np.zeros_like(times)
        self.com_coords = []
        self.com_obs = []
        for i, t in enumerate(times):
            # print("In simulator time: ", t)
            com = self.rob.step(t, duration=times[i] - times[i - 1])
            self.coords[i, :] = copy.deepcopy(self.rob.coordinate)
            self.obs[i] = copy.deepcopy(self.rob.current_observation)
            if com is not None:
                self.com_coords.append(com[0])
                self.com_obs.append(com[1])

        self.com_coords = np.asarray(self.com_coords)
        self.com_obs = np.asarray(self.com_obs)

        if self.ref_global:
            # Convert to global latlon
            self.global_coords = convert_to_latlon(self.coords, self.env.extent.origin)
            self.global_com_coords = convert_to_latlon(self.com_coords, self.env.extent.origin)

    def plot_comms(self, filename="comm_data"):
        """Plot robot communicated signal results."""
        if self.com_coords is None or self.com_obs is None:
            raise ValueError("Must run simluation first to plot.")

        # [TODO] need better vector-obs visualization handling
        if type(self.com_obs[0]) is list and len(self.com_obs[0]) > 0:
            self.com_obs = [o[-1] for o in self.com_obs]

        if self.ref_global:
            coords = self.global_com_coords
        else:
            coords = self.com_coords

        plt.scatter(coords[:, 0], coords[:, 1], c=self.com_obs,
                    s=1, cmap='viridis', vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs))
        plt.axis('square')
        if self.experiment_name is not None:
            filename = f"{self.experiment_name}_{filename}"
        fpath = os.path.join(output_home(), f'{filename}.png')
        plt.savefig(fpath)

    def plot_all(self, filename="all_data"):
        """Plot all robot data."""
        if self.com_coords is None or self.com_obs is None:
            raise ValueError("Must run simluation first to plot.")

        # [TODO] need better vector-obs visualization handling
        if type(self.obs[0]) is list and len(self.obs[0]) > 0:
            self.obs = [o[-1] for o in self.obs]
        if self.experiment_name is not None:
            filename = f"{self.experiment_name}_{filename}"

        if self.ref_global:
            coords = self.global_coords
        else:
            coords = self.coords
        plt.scatter(coords[:, 0], coords[:, 1], c=self.obs, s=1,
                    cmap='viridis', vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs))
        plt.axis('square')
        fpath = os.path.join(output_home(), f'{filename}.png')
        plt.savefig(fpath)

    def plot_world(self, filename='simulation'):
        """Plot results w.r.t. world and save as a GIF."""
        print("Generating global simulation.")
        if self.experiment_name is not None:
            filename = f"{self.experiment_name}_{filename}"

        frame_skip = 600  # number of seconds between each frame (2.5 mins)
        time_frames = list(enumerate(self.times))[::frame_skip]

        fig, ax = plt.subplots(1, 1)
        ax.axis('equal')
        ax.axis('off')

        x = np.linspace(self.env.extent.xmin,
                        self.env.extent.xmax,
                        self.env.extent.xres)
        y = np.linspace(self.env.extent.ymin,
                        self.env.extent.ymax,
                        self.env.extent.yres)
        z = np.linspace(self.env.extent.zmin,
                        self.env.extent.zmax,
                        self.env.extent.zres)
        xyz = np.hstack([x, y, z])

        if self.ref_global:
            # Convert to global latlon
            xyz = convert_to_latlon(xyz, self.env.extent.origin)
            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]

        xm, ym = np.meshgrid(x, y)

        def animate(index):
            i = time_frames[index][0]
            t = time_frames[index][1]

            # Get the most recent snapshot
            snap = self.env.get_snapshot(t)

            cf = ax.contourf(xm, ym, snap,
                             vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs))

            # Scatter the points in red
            if self.ref_global:
                coords = self.global_coords
            else:
                coords = self.coords
            s1 = ax.scatter(coords[:i, 0], coords[:i, 1], c='r', s=20,
                            vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs))

            # Overlay sample values
            s2 = ax.scatter(coords[:i, 0], coords[:i, 1],
                            c=self.obs[:i], cmap='viridis', s=10, alpha=0.5,
                            vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs))
            return cf, s1, s2

        tic()
        anim = animation.FuncAnimation(fig=fig, func=animate,
                                       frames=len(time_frames))

        fpath = os.path.join(output_home(), f'{filename}.gif')
        anim.save(fpath, writer='imagemagick', fps=10)
        toc()

    def plot_world3d(self, times=None, filename='simulation3d', bathy_file=None):
        """Plot results w.r.t. world and save as a GIF."""
        print("Generating 3d simulation.")
        if self.experiment_name is not None:
            filename = f"{self.experiment_name}_{filename}"

        if times is None:
            times = self.times

        if self.ref_global:
            coords = self.global_coords
        else:
            coords = self.coords

        if bathy_file is None:
            bathy_file = os.path.join(
                os.getenv("SENTRY_DATA"), f"processed/guaymas_bathy.csv")

        with open(bathy_file, "rb") as fh:
            bathy = pd.read_csv(fh)

        sites_file = os.path.join(os.getenv("SENTRY_DATA"), f"site_data.yaml")
        sites = yaml.load(open(sites_file), Loader=yaml.FullLoader)
        site_locs = np.array([(x['lon'], x['lat'], 0.0) for x in sites.values()]).reshape(-1, 3)
        # Find site depth from nearest bathy point
        for i in range(site_locs.shape[0]):
            d = bathy.iloc[np.abs(bathy[['long', 'lat']] - site_locs[i, 0:2]
                                  ).sum(axis=1).idxmin()]['depth']
            site_locs[i, 2] = -d

        fig1 = scatter_plume_and_traj(
            times, self.env, self.times, coords, bathy, site_locs,
            title="Plume Evolution", xlim=[-2000, 2000],
            ylim=[-2000, 2000], zlim=[-300, 150], ref_global=self.ref_global)

        fig2 = plot_window_in_bathy(self.env, bathy, site_locs, xlim=[-2000, 2000],
                                    ylim=[-2000, 2000], zlim=[-300, 150])
        return fig1, fig2
