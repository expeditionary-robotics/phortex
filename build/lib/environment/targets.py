'''This file provides several classes of moving target object for spoofing'''
import numpy as np
import matplotlib.pyplot as plt

from .environment import Environment


class Bullseye(Environment):
    '''Creates a single moving target according to an input xy trajectory'''

    def __init__(self, extent, xcoord, ycoord, l=1.5, A=1.5):
        '''Initialize Bullseye environment.

        Args:
            xcoord (fun): function of center in x-axis in time
            ycoord (fun): function of center in y-axis in time
            l (float): lengthscale as a float
            A (float): maximum value as a float
        '''
        self.extent = extent
        self.cx, self.cy = xcoord, ycoord
        self.xm, self.ym = np.meshgrid(
            np.linspace(self.extent.xmin, self.extent.xmax, self.extent.xres),
            np.linspace(self.extent.ymin, self.extent.ymax, self.extent.yres))
        self.l = l
        self.A = A

    def get_snapshot(self, t):
        """Get a ground truth full state at time t.

                Args:
                        t (float): the global time

                Returns: (np.array) ground truth bullseye values, nx x ny array
                """
        cx, cy = self.cx(t), self.cy(t)
        snapshot = self.A * \
            np.exp(-(self.l**2 * ((self.xm - cx)**2 + (self.ym - cy)**2)))
        return snapshot

    def get_value(self, t, loc):
        """Get ground truth at time t and location loc.

                Args:
                        t (float): the global time
                        loc (tuple[float]): a location, in xy space

                Returns (float): the concentration value
                """
        x = loc[0]
        y = loc[1]
        cx, cy = self.cx(t), self.cy(t)
        val = self.A * np.exp(-(self.l**2 * ((x - cx)**2 + (y - cy)**2)))
        return val

    def get_maxima(self, t):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time

        Returns: tuple[float] maximum location
        """
        # Get snapshot
        cx, cy = self.cx(t), self.cy(t)
        val = self.A * \
            np.exp(-(self.l**2 * ((self.xm - cx)**2 + (self.ym - cy)**2)))
        return (cx, cy)


if __name__ == '__main__':
    target = Bullseye(nx=100, ny=100, xscale=2, yscale=2,
                      xcoord=lambda t: 0.5 * np.sin(t) + 1, ycoord=lambda t: 0.5 * np.cos(t) + 1)
    snaps = np.linspace(0, 2 * np.pi, 5)
    for t in snaps:
        plt.imshow(target.get_snapshot(t))
        plt.show()

    vals = target.get_value(snaps[1], np.array(
        [1.0, 1.0]), np.array([1.0, 1.0]))
    print(vals)
