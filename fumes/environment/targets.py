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

    def get_snapshot(self, t, **kwargs):
        """Get a ground truth full state at time t.

                Args:
                        t (float): the global time

                Returns: (np.array) ground truth bullseye values, nx x ny array
                """
        cx, cy = self.cx(t), self.cy(t)
        snapshot = self.A * \
            np.exp(-(self.l**2 * ((self.xm - cx)**2 + (self.ym - cy)**2)))
        return snapshot

    def get_value(self, t, loc, **kwargs):
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

    def get_maxima(self, t, **kwargs):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time

        Returns: tuple[float] maximum location
        """
        # Get snapshot
        return (self.cx(t), self.cy(t))


class DynamicBullseye(Bullseye):
    '''Creates a single moving target that changes size according to an
    input xy trajectory'''

    def __init__(self, extent, xcoord, ycoord, lcoord, thcoord, l=1.5, A=1.5):
        '''Initialize Bullseye environment.

        Args:
            xcoord (fun): function of center in x-axis in time
            ycoord (fun): function of center in y-axis in time
            lcoord (fun): function of the lengthscale scaling in time
            thcoord (fun): function of target orientation in time
            l (float): lengthscale scaling
            A (float): maximum value as a float
        '''
        super().__init__(extent, xcoord, ycoord, l, A)
        self.l_t = lcoord
        self.th_t = thcoord

    def _get(self, t, xlocs, ylocs, **kwargs):
        """Get a ground truth full state at time t.

        Args:
            t (float): the global time

        Returns: (np.array) ground truth bullseye values, nx x ny array
        """
        cx, cy = self.cx(t), self.cy(t)
        l_t, th_t = self.l_t(t, self.l), self.th_t(t)

        """Compute Gaussian paramters using equations from:
        https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
        """
        l_x = self.l
        l_y = 2.0 * l_t
        a = np.cos(th_t)**2 / (2 * l_x**2) + np.sin(th_t)**2 / (2 * l_y**2)
        b = -np.sin(2 * th_t) / (4 * l_x**2) + np.sin(2 * th_t) / (4 * l_y**2)
        c = np.sin(th_t)**2 / (2 * l_x**2) + np.cos(th_t)**2 / (2 * l_y**2)

        snapshot = self.A * \
            np.exp(-(a * (xlocs - cx)**2 + 2 * b * (xlocs - cx) *
                   (ylocs - cy) + c * (ylocs - cy)**2))
        return snapshot

    def get_snapshot(self, t, **kwargs):
        """Get a ground truth full state at time t.

        Args:
            t (float): the global time

        Returns: (np.array) ground truth bullseye values, nx x ny array
        """
        # import pdb
        # pdb.set_trace()
        return self._get(t, self.xm, self.ym)

    def get_value(self, t, loc, **kwargs):
        """Get ground truth at time t and location loc.

        Args:
                t (float): the global time
                loc (tuple[float]): a location, in xy space

        Returns (float): the concentration value
        """
        x = loc[0]
        y = loc[1]
        return self._get(t, x, y)

    def get_maxima(self, t, **kwargs):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time

        Returns: tuple[float] maximum location
        """
        # Get snapshot
        return (self.cx(t), self.cy(t))


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
