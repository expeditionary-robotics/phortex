"""Defines a lawnmower Trajectory with a start point that differs from the
origin parameter."""
import copy
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
from shapely import affinity

from .trajectory import Trajectory
from .utils import distance
from .lawnmower import Lawnmower, Spiral


class LawnmowerWithStart(Lawnmower):
    """Creates a typical Boustrophedon path and a start point."""

    def __init__(self, t0, vel, lh, lw, resolution, altitude=50., origin=(0, 0),
                 orientation=0., start_point=None, noise=None):
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
            start_point(tuple[float]): start point of the trajectory. If None,
                defaults to the origin
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

        if start_point is None:
            self.start_point = copy.copy(origin)
        else:
            self.start_point = start_point

        self._create_global_frame()

        self._length = None

    def _json_stats(self):
        """Returns a dict of trajectory information."""
        json_dict = {"t0": float(self.t0),
                     "vel": self.vel,
                     "lh": self.lh,
                     "lw": self.lw,
                     "resolution": self.resolution,
                     "noise": self.noise,
                     "origin": self.origin,
                     "orientation": self.orientation,
                     "altitude": self.altitude,
                     "start_point": self.start_point}
        return json_dict

    def _create_global_frame(self):
        """Creates the lawnmower grid points w/start in the global frame"""
        # Generate the lawnmower global frame
        super()._create_global_frame()

        # Create a line from the start point to the origin
        line_coords = [(self.start_point[0], self.start_point[1]),
                       (self.origin[0], self.origin[1])]

        all_coords_x = np.hstack(
            [np.array([self.start_point[0], self.origin[0]]).flatten(),
                self.path.xy[0]])
        all_coords_y = np.hstack(
            [np.array([self.start_point[1], self.origin[1]]).flatten(),
             self.path.xy[1]])
        self.path = LineString(zip(all_coords_x, all_coords_y))


class SpiralWithStart(Spiral):
    """Creates a spiral path."""

    def __init__(self, t0, vel, lh, lw, resolution, altitude=50., origin=(0, 0),
                 orientation=0., start_point=None, noise=None):
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
            start_point(tuple[float]): start point of the trajectory. If None,
                defaults to the origin
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

        if start_point is None:
            self.start_point = copy.copy(origin)
        else:
            self.start_point = start_point

        self._create_global_frame()

        self._length = None

    def _json_stats(self):
        """Returns a dict of trajectory information."""
        json_dict = {"t0": float(self.t0),
                     "vel": self.vel,
                     "lh": self.lh,
                     "lw": self.lw,
                     "resolution": self.resolution,
                     "noise": self.noise,
                     "origin": self.origin,
                     "orientation": self.orientation,
                     "altitude": self.altitude,
                     "start_point": self.start_point}
        return json_dict

    def _create_global_frame(self):
        """Creates the lawnmower grid points w/start in the global frame"""
        # Generate the lawnmower global frame
        super()._create_global_frame()

        # Create a line from the start point to the origin
        line_coords = [(self.start_point[0], self.start_point[1]),
                       (self.origin[0], self.origin[1])]

        all_coords_x = np.hstack(
            [np.array([self.start_point[0], self.origin[0]]).flatten(),
                self.path.xy[0]])
        all_coords_y = np.hstack(
            [np.array([self.start_point[1], self.origin[1]]).flatten(),
             self.path.xy[1]])
        self.path = LineString(zip(all_coords_x, all_coords_y))
