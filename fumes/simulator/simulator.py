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

    def __init__(self, robot, environment, ref_global=False, reward=None):
        """Initialize simulator.

        Args:
            robot (Robot): a Robot object
            environment (Environment): an Environment object
            ref_global (bool): if True, simulation plots are
                generated in a global reference frame.
            reward (Reward): a reward object the encodes the task
                Used to compute summary statistics
        """
        self.rob = robot
        self.env = environment
        self.ref_global = ref_global  # whether to plot in global reference
        self.reward = reward  # the reward object encoding the task
        self.coords = None  # coordinates visited by robot
        self.obs = None  # observations taken by robot
        self.com_coords = None  # coordinates communicated online
        self.com_obs = None  # observations communicated online
        self.with_noise = None  # whether simulator returns noisy data
        self.noise_proportion = None  # proportion of simulator obs corrupted

    def simulate(self, times, experiment_name=None, with_noise=False, noise_portion=0.1):
        """Perform simulation.

        Args:
            times (np.array): array of times to simulate
            experiment_name (string): exp name running this simulator
            with_noise (bool): whether to corrupt simulated observations with noise
            noise_portion (float): percentage corrupted noise
        """
        self.experiment_name = experiment_name
        self.with_noise = with_noise
        self.noise_proportion = noise_portion
        self.times = times
        self.coords = np.zeros((len(times), 3)).astype(float)
        self.obs = np.zeros_like(times).astype(float)
        self.com_coords = []
        self.com_obs = []
        for i, t in enumerate(times):
            # print("In simulator time: ", t)
            com = self.rob.step(t, duration=times[i] - times[i - 1])
            self.coords[i, :] = copy.deepcopy(self.rob.coordinate)
            if with_noise is True:
                obs = copy.deepcopy(self.rob.current_observation)
                choice = np.random.choice([0,1],1, replace=True, p=[noise_portion, 1-noise_portion])
                self.obs[i] = np.log((1. * choice) + (obs))[0]
            else:
                self.obs[i] = np.log(1. + copy.deepcopy(self.rob.current_observation))[0]
            if com is not None:
                self.com_coords.append(com[0])
                self.com_obs.append(com[1])
        self.com_coords = np.asarray(self.com_coords)
        self.com_obs = np.asarray(self.com_obs)

        if self.ref_global:
            # Convert to global latlon
            self.global_coords = convert_to_latlon(self.coords, self.env.extent.origin)
            self.global_com_coords = convert_to_latlon(self.com_coords, self.env.extent.origin)

    def _json_stats(self):
        """Generate summary statistics about the simulation."""

        json_dict = {"total_obs": len(self.obs),
                     "total_com_obs": len(self.com_obs),
                     "obs": self.obs.tolist(),
                     "coords": self.coords.tolist(),
                     "com_obs": self.com_obs.tolist(),
                     "com_coords": self.com_coords.tolist(),
                     "times": self.times.tolist(),
                     "sim_with_noise": self.with_noise,
                     "noise_proportion": self.noise_proportion,
                     }
        if self.ref_global:
            json_dict["ref_global"] = self.ref_global
            json_dict["global_coords"] = self.global_coords.tolist(),
            json_dict["global_com_coords"] = self.global_com_coords.tolist()

        return json_dict

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

    def plot_world(self, filename='simulation', frame_skip=300):
        """Plot results w.r.t. world and save as a GIF."""
        print("Generating global simulation.")
        if self.experiment_name is not None:
            filename = f"{self.experiment_name}_{filename}"

        frame_skip = frame_skip  # number of seconds between each frame
        time_frames = list(enumerate(self.times))[::frame_skip]

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

        # Initialize lists of locations
        xm, ym = np.meshgrid(x, y)

        # Instantiate the animation plot
        fig, ax = plt.subplots(1, 1)
        ax.set_xlim([self.env.extent.xmin, self.env.extent.xmax])
        ax.set_ylim([self.env.extent.ymin, self.env.extent.ymax])
        ax.axis('equal')
        ax.axis('off')

        def init():
            global cf
            global s1
            global s2

            i = time_frames[0][0]
            t = time_frames[0][1]

            # Get the most recent snapshot
            snap = self.env.get_snapshot(t)

            cf = ax.contourf(xm, ym, snap,
                             vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs),
                             zorder=0)
            ax.set_xlim([self.env.extent.xmin, self.env.extent.xmax])
            ax.set_ylim([self.env.extent.ymin, self.env.extent.ymax])

            # Scatter the points in red
            if self.ref_global:
                coords = self.global_coords
            else:
                coords = self.coords
            s1 = ax.scatter(coords[:i, 0], coords[:i, 1], c='r', s=20,
                            vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs),
                            zorder=10)

            # Overlay sample values
            s2 = ax.scatter(coords[:i, 0], coords[:i, 1],
                            c=self.obs[:i], cmap='viridis', s=2, alpha=0.5,
                            vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs),
                            zorder=20)
            return [s1, s2, cf]

        def animate(index):
            global cf
            global s1
            global s2

            i = time_frames[index][0]
            t = time_frames[index][1]

            # Get the most recent snapshot
            snap = self.env.get_snapshot(t)

            for c in cf.collections:
                c.remove()  # removes only the contours, leaves the rest intact

            cf = ax.contourf(xm, ym, snap,
                             vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs),
                             zorder=0)

            # Scatter the points in red
            if self.ref_global:
                coords = self.global_coords
            else:
                coords = self.coords
            s1.set_offsets(coords[:i, :])

            s2.set_offsets(coords[:i, :])
            s2.set_array(self.obs[:i])
            return [cf, s1, s2]

        print("Starting animation...")
        tic()
        anim = animation.FuncAnimation(fig=fig, func=animate, init_func=init,
                                       frames=len(time_frames), repeat=True)

        fpath = os.path.join(output_home(), f'{filename}.gif')
        plt.rcParams["animation.convert_path"] = r'/usr/bin/convert'
        # anim.save(fpath, writer='imagemagick', fps=10)
        anim.save(fpath, writer=animation.FFMpegWriter())
        toc()

    def plot_world_hpc(self, filename='simulation', frame_skip=300):
        """Plot results w.r.t. world and save as a GIF."""
        print("Generating global simulation.")
        if self.experiment_name is not None:
            filename = f"{filename}"

        time_frames = list(enumerate(self.times))[::frame_skip]

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

        # Initialize lists of locations
        xm, ym = np.meshgrid(x, y)
        zref = self.coords[0][-1]

        tic()
        for i in range(len(time_frames)):
            t = time_frames[i][1]
            j = time_frames[i][0]
            fig, ax = plt.subplots(1, 1)
            ax.set_xlim([self.env.extent.xmin, self.env.extent.xmax])
            ax.set_ylim([self.env.extent.ymin, self.env.extent.ymax])
            ax.axis('equal')
            ax.axis('off')
            snap = self.env.get_snapshot(t, z=[zref])
            cf = ax.contourf(xm, ym, snap[0], vmin=np.nanmin(self.obs),
                             vmax=np.nanmax(self.obs), zorder=0)
            if self.ref_global:
                coords = self.global_coords
            else:
                coords = self.coords
            s1 = ax.scatter(coords[:j, 0], coords[:j, 1], c='r', s=20,
                            vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs),
                            zorder=10)

            # Overlay sample values
            s2 = ax.scatter(coords[:j, 0], coords[:j, 1],
                            c=self.obs[:j], cmap='viridis', s=2, alpha=0.5,
                            vmin=np.nanmin(self.obs), vmax=np.nanmax(self.obs),
                            zorder=20)
            fpath = os.path.join(f'{filename}_frame{i}_time{t}.png')
            plt.savefig(fpath)
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
