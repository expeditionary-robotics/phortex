"""Defines a lawnmower Trajectory object that dynamically accepts parameters."""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely import affinity

from .trajectory import Trajectory
from .utils import distance

import pdb


class Lawnmower(Trajectory):
    """Creates a typical Boustrophedon path."""

    def __init__(self, t0, vel, lh, lw, resolution, altitude=50., origin=(0, 0),
                 orientation=0., noise=None):
        """ Initialize the static parameters of lawnmower.

        Args:
            t0 (float): initial time
            vel (float): estimated velocity
            lh (float): total path height
            lw (float): total path width
            resolution (float): smallest leg distance
            altitude (float): height at which this trajectory is planned
            noise (float): perturbations added to trajectory
            origin(tuple[float]): starting point of the lawnmower
            orientation(float): starting angle of the lawnmower w.r.t.
               global coordinates
        """
        # Travel parameters
        self.t0 = t0
        self.vel = vel

        # Local reference frame
        self.lh = lh
        self.lw = lw
        self.resolution = resolution
        self.noise = noise
        self.origin = origin
        self.orientation = orientation
        self.altitude = altitude

        self._create_global_frame()

        self._length = None

    def _create_local_frame(self):
        """Creates the lawnmower grid points in the local frame."""
        num_passes = int(self.lh / self.resolution)
        ypatt = np.array([[i * self.resolution, i * self.resolution]
                         for i in range(num_passes + 1)]).flatten()
        xpatt = []
        patt = [0, self.lw, self.lw, 0]
        for i in range(ypatt.shape[0]):
            xpatt.append(patt[i % 4])
        xpatt = np.array(xpatt)

        self.xcoords = xpatt
        self.ycoords = ypatt
        self.coords = [(x, y) for x, y in zip(self.xcoords, self.ycoords)]
        self.path = LineString(self.coords)
        self.corners = LineString([
            (0., 0.),
            (0., self.lh),
            (self.lw, 0.),
            (self.lw, self.lh)
        ])

    def _create_global_frame(self):
        """Creates the lawnmower grid points onto the global frame"""
        self._create_local_frame()

        # Translate trajectory
        self.path = affinity.translate(
            self.path, self.origin[0], self.origin[1])
        self.corners = affinity.translate(
            self.corners, self.origin[0], self.origin[1])

        # Rotate trajectory
        self.path = affinity.rotate(
            self.path, self.orientation, origin=self.origin)
        self.corners = affinity.rotate(
            self.corners, self.orientation, origin=self.origin)

    @property
    def length(self):
        if self._length is None:
            self._length = self.path.length
        return self._length

    @property
    def time_at_end(self):
        return self.t0 + self.length / self.vel

    @property
    def xmin(self):
        return min(np.array(self.corners.xy)[0, :])

    @property
    def xmax(self):
        return max(np.array(self.corners.xy)[0, :])

    @property
    def ymin(self):
        return min(np.array(self.corners.xy)[1, :])

    @property
    def ymax(self):
        return max(np.array(self.corners.xy)[1, :])


class Spiral(Lawnmower):
    """Creates a spiral path."""

    def __init__(self, t0, vel, lh, lw, resolution, altitude=50., origin=(0, 0),
                 orientation=0., noise=None):
        """ Initialize the static parameters of spiral.

        Args:
            t (float): initial time
            vel (float): estimated velocity
            lh (float): total path height
            lw (float): total path width
            resolution (float): smallest leg distance
            altitude (float): height at which this trajectory is planned
            origin(tuple[float]): starting point of the lawnmower
            orientation(float): starting angle of the lawnmower w.r.t.
               global coordinates
            noise (float): perturbations added to trajectory
        """
        # Call Lawnmower constructor
        super().__init__(t0, vel, lh, lw, resolution, altitude, origin,
                         orientation, noise)

    def _create_local_frame(self):
        """Creates the spiral grid points in the local frame."""
        num_rings = int(min(self.lh, self.lw) /
                        self.resolution)  # get the number of rings possible
        # get the minimum distance for longer leg
        res_scaled = max(self.lh, self.lw) / num_rings
        ypatt = [1., -1., -1., 1.]
        xpatt = [1., 1., -1., -1.]
        xcoords = [0]
        ycoords = [0]
        for r in range(num_rings):
            if self.lw > self.lh:
                for i, j in zip(xpatt, ypatt):
                    xcoords.append(r * i * res_scaled)
                    ycoords.append(r * j * self.resolution)
            else:
                for i, j in zip(xpatt, ypatt):
                    xcoords.append(r * i * self.resolution)
                    ycoords.append(r * j * res_scaled)

        self.xcoords = np.asarray(xcoords)
        self.ycoords = np.asarray(ycoords)
        coords = [(x, y) for x, y in zip(self.xcoords, self.ycoords)]
        self.path = LineString(coords)
        self.corners = LineString([
            (0., 0.),
            (0., self.lh),
            (self.lw, 0.),
            (self.lw, self.lh)
        ])
