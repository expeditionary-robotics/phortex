"""Utility for generating trajectory objects."""
from abc import ABC, abstractmethod
import numpy as np

from fumes.trajectory import Lawnmower, Spiral,  \
    LawnmowerWithStart, SpiralWithStart


class TrajectoryGenerator(ABC):
    """Trajectory generation from parameters."""

    def __init__(self, t0, vel, alt):
        """Initializee parameters."""
        self.t0 = t0  # initial time
        self.vel = vel  # velocity
        self.alt = alt

    @abstractmethod
    def generate(self):
        """Generate trajectory object."""
        pass


class LawnSpiralGenerator(TrajectoryGenerator):
    """Trajectory geneartor for Lawnmower or Spiral trajectories."""

    def __init__(self, t0, vel, alt, traj_type, lh=500, lw=500, resolution=10):
        """Initialize trajectory generator parameters.

        Args:
            t0 (float): initial time
        """
        super().__init__(t0, vel, alt)
        self.traj_type = traj_type
        self.lh = l

        self.lw = lw
        self.resolution = resolution

    def generate(self, orientation, origin_x, origin_y):
        if self.traj_type == "lawnmower":
            return Lawnmower(self.t0, self.vel, self.lh, self.lw,
                             self.resolution, noise=0.0,
                             orientation=orientation,
                             origin=(origin_x, origin_y),
                             altitude=self.alt)
        elif self.traj_type == "spiral":
            return Spiral(self.t0, self.vel, self.lh, self.lw, self.resolution,
                          noise=0.0, orientation=orientation,
                          origin=(origin_x, origin_y),
                          altitude=self.alt)

class LawnSpiralAltGenerator(TrajectoryGenerator):
    """Trajectory geneartor for Lawnmower or Spiral trajectories."""

    def __init__(self, t0, vel, traj_type, lh=500, lw=500, resolution=10):
        """Initialize trajectory generator parameters.

        Args:
            t0 (float): initial time
        """
        super().__init__(t0, vel, 0.0)
        self.traj_type = traj_type
        self.lh = l
        
        self.lw = lw
        self.resolution = resolution

    def generate(self, orientation, origin_x, origin_y, alt):
        if self.traj_type == "lawnmower":
            return Lawnmower(self.t0, self.vel, self.lh, self.lw,
                             self.resolution, noise=0.0,
                             orientation=orientation,
                             origin=(origin_x, origin_y),
                             altitude=alt)
        elif self.traj_type == "spiral":
            return Spiral(self.t0, self.vel, self.lh, self.lw, self.resolution,
                          noise=0.0, orientation=orientation,
                          origin=(origin_x, origin_y),
                          altitude=alt)



class LawnSpiralGeneratorFlexible(TrajectoryGenerator):
    def __init__(self, t0, vel, alt, traj_type, res=10.):
        super().__init__(t0, vel, alt)
        self.traj_type = traj_type
        self.res = res

    def generate(self, lh, lw, orientation, origin_x, origin_y):
        if self.traj_type == "lawnmower":
            return Lawnmower(self.t0, self.vel, lh, lw,
                             self.res,
                             noise=0.0,
                             orientation=orientation,
                             origin=(origin_x, origin_y),
                             altitude=self.alt)
        elif self.traj_type == "spiral":
            return Spiral(self.t0, self.vel, self.lh, self.lw, self.res, noise=0.0,
                          orientation=orientation,
                          origin=(origin_x, origin_y),
                          altitude=self.alt)

class LawnSpiralAltGeneratorFlexible(TrajectoryGenerator):
    def __init__(self, t0, vel, traj_type, res=10.):
        super().__init__(t0, vel, 0.0)
        self.traj_type = traj_type
        self.res = res

    def generate(self, lh, lw, orientation, origin_x, origin_y):
        if self.traj_type == "lawnmower":
            return Lawnmower(self.t0, self.vel, lh, lw,
                             self.res,
                             noise=0.0,
                             orientation=orientation,
                             origin=(origin_x, origin_y),
                             altitude=alt)
        elif self.traj_type == "spiral":
            return Spiral(self.t0, self.vel, self.lh, self.lw, self.res, noise=0.0,
                          orientation=orientation,
                          origin=(origin_x, origin_y),
                          altitude=alt)


class LawnSpiralWithStartGeneratorFlexible(TrajectoryGenerator):
    def __init__(self, t0, vel, alt, start_point, traj_type, res=10.):
        super().__init__(t0, vel, alt)
        self.traj_type = traj_type
        self.res = res
        self.start_point = start_point
        # self.lh = lh
        # self.lw = lw

    def generate(self, lh, lw, orientation, origin_x, origin_y):
        if self.traj_type == "lawnmower":
            return LawnmowerWithStart(self.t0, self.vel, lh, lw,
                                      self.res,
                                      noise=0.0,
                                      orientation=orientation,
                                      origin=(origin_x, origin_y),
                                      start_point=self.start_point,
                                      altitude=self.alt)
        elif self.traj_type == "spiral":
            return SpiralWithStart(self.t0, self.vel, self.lh, self.lw,
                                   self.res, noise=0.0,
                                   orientation=orientation,
                                   origin=(origin_x, origin_y),
                                   start_point=self.start_point,
                                   altitude=self.alt)

class LawnSpiralAltWithStartGeneratorFlexible(TrajectoryGenerator):
    def __init__(self, t0, vel, start_point, traj_type, res=10.):
        super().__init__(t0, vel, 0.0)
        self.traj_type = traj_type
        self.res = res
        self.start_point = start_point
        # self.lh = lh
        # self.lw = lw

    def generate(self, lh, lw, orientation, origin_x, origin_y, alt):
        if self.traj_type == "lawnmower":
            return LawnmowerWithStart(self.t0, self.vel, lh, lw,
                                      self.res,
                                      noise=0.0,
                                      orientation=orientation,
                                      origin=(origin_x, origin_y),
                                      start_point=self.start_point,
                                      altitude=alt)
        elif self.traj_type == "spiral":
            return SpiralWithStart(self.t0, self.vel, self.lh, self.lw,
                                   self.res, noise=0.0,
                                   orientation=orientation,
                                   origin=(origin_x, origin_y),
                                   start_point=self.start_point,
                                   altitude=alt)
