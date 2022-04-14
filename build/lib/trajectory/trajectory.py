"""Abstract base class for a Trajectory object."""

from abc import ABC, abstractmethod
import numpy as np

from .utils import distance

import pdb


class Trajectory(ABC):
    """Trajectory base class."""

    @abstractmethod
    def __init__(self, t0, vel, *args, **kwargs):
        """Initialize and populate trajectory.

        Args:
            t0 (float): intial time
            vel (float): estimated velocity
        """
        pass

    def plot(self, ax=None, title=None):
        """Plot trajectory, optionally on provided axis object."""
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        ax.plot(*self.path.xy)

        if title is not None:
            ax.set_title(title)

        return ax

    def uniformly_sample(self, samp_dist):
        """Generate samples on the trajectory at a given resolution.

        Args:
            samp_dist(float): distance between samples

        Returns: list[tuple(float, float, float)]: an list of sample locations,
            where each list member is a tuple of (time, x, y)
        """
        path_dist = sum(distance(
            self.xcoords[i],
            self.ycoords[i],
            self.xcoords[i + 1],
            self.ycoords[i + 1]) for i in range(self.xcoords.shape[0] - 1))

        num_samples = int(path_dist / samp_dist)
        samples = []
        for j in range(num_samples):
            dist_on_traj = samp_dist * j
            samples.append(self.path_sample(dist_on_traj))
        return samples

    def path_sample(self, dist_on_traj):
        """Generate samples at a given distance along the trajectory.

        Args:
            dist_on_traj (float): distance along trajectory

        Returns: tuple(float, float, float, float)]: a sample location, represetned as
            a tuple of (time, x, y, z)
        """
        # Generate lawnmower points in global frame
        loc = list(self.path.interpolate(dist_on_traj).coords)[0]
        time = self.t0 + dist_on_traj / self.vel
        coord = (time, loc[0], loc[1], self.altitude)
        if self.noise is not None:
            coord = (coord[0],                                      # time
                     coord[1] + np.random.normal(scale=self.noise),  # x
                     coord[2] + np.random.normal(scale=self.noise),  # y
                     coord[3] + np.random.normal(scale=self.noise))  # z
        return coord

    @property
    @abstractmethod
    def length(self):
        """Length of the trajectory."""
        pass

    @property
    @abstractmethod
    def xmin(self):
        """Minimum x coordinate of the trajectory."""
        pass

    @property
    @abstractmethod
    def xmax(self):
        """Maximum x coordinate of the trajectory."""
        pass

    @property
    @abstractmethod
    def ymin(self):
        """Minimum y coordinate of the trajectory."""
        pass

    @property
    @abstractmethod
    def ymax(self):
        """Maximum y coordinate of the trajectory."""
        pass
