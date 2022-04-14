"""Defines a chain of trajectories, to be executed in sequence."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely import affinity

from .trajectory import Trajectory
from .utils import distance

import pdb


class Chain(Trajectory):
    """Creates a chain of trajectory objects."""

    def __init__(self, t0, vel, traj_list, altitude=[50.], noise=None):
        """ Initialize the static parameters of lawnmower.

        Args:
            t0 (float): initial time
            vel (float): estimated velocity
            traj_list (list[Trajectory]): list of trajectory objects to
                include in the chain
            altitude (float): height at which this trajectory is planned
            noise (float): perturbations added to trajectory
        """
        # Travel parameters
        self.t0 = t0
        self.vel = vel

        # [TODO] right now, a chain can only have a single altitude
        self.altitude = altitude

        # Trajectory  info
        self.noise = noise
        self.traj_list = traj_list
        self._length = None

        self._create_global_frame()

    def _create_global_frame(self):
        """Creates the chain grid points in the global frame"""
        all_coords_x = []
        all_coords_y = []
        all_coords_z = []
        for i, traj in enumerate(self.traj_list):
            all_coords_x += traj.path.xy[0]
            all_coords_y += traj.path.xy[1]
            # all_coords_z += [self.altitude[i]]*len(traj.path.xy[0])
        self.path = LineString(zip(all_coords_x, all_coords_y))
        self.xcoords = np.hstack([traj.xcoords for traj in self.traj_list])
        self.ycoords = np.hstack([traj.ycoords for traj in self.traj_list])
        # self.zcoords = all_coords_z

    @property
    def length(self):
        if self._length is None:
            self._length = sum([traj.length for traj in self.traj_list])
        return self._length

    @property
    def xmin(self):
        return min([traj.xmin for traj in self.traj_list])

    @property
    def xmax(self):
        return max([traj.xmax for traj in self.traj_list])

    @property
    def ymin(self):
        return min([traj.ymin for traj in self.traj_list])

    @property
    def ymax(self):
        return max([traj.ymax for traj in self.traj_list])
