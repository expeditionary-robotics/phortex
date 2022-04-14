"""Fully observeable 2D model."""

from .model import Model


class FullyObs(Model):
    def __init__(self, extent, environment):
        """Initialize  fully observeable model.

        Args:
            extent (Extent): a geographic Extent object
            environment (Environment): an Environment object
        """
        self.extent = extent
        self.env = environment

    def get_value(self, t, loc):
        """Get a deterministic prediction of the model output t and loc.

                Args:
                        t (float): the global time
                        loc (tuple[float]): a location, in xy or xyz space

                Returns (float): the concentration value
                """
        return self.env.get_value(t, loc)

    def get_prediction(self, t, loc):
        """Get a (mean, variance) prediction of the model output t and loc.

                Args:
                        t (float): the global time
                        loc (tuple[float]): a location, in xyz or xy space

                Returns:
                        tuple(float, float): mean and variance of concentration value
                """
        return self.env.get_value(t, loc), 0.0

    def get_snapshot(self, t):
        """Get a deterministic prediction of full state at time t.

                Args:
                        t (float): the global time

                Returns:
                        np.array: full state snapshot at time t
                """
        return self.env.get_snapshot(t)

    def get_snapshot_prediction(self, t):
        """Get a deterministic prediction of full state and uncertainty at t.

                Args:
                        t (float): the global time

                Returns:
                        np.array: mean state snapshot
                        np.array: predicted variance of state snapshot
                """
        snapshot = self.env.get_snapshot(t)
        variance = np.zeros(snapshot.shape)
        return snapshot, variance

    def get_maxima(self, t):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time

        Returns: list[float tuple] maximum location
        """
        # Get snapshot
        return self.env.get_maxima(t)

    def update(self, t, loc, obs):
        """Update model with observation at t, loc.

                Args:
                        t (float): the global time
                        loc (tuple[float]): a location, in xyz or xy space
                        obs (tuple[float]): a sensor observation
                """
        pass

    def update_multiple(self, t, loc, obs):
        """Update model with a list of observations at t, loc.

        Args:
            t (list[float]): the global time
            loc (list[tuple[float]]): a list of locations, in xyz or xy space
            obs (list[tuple[float]]): a list of sensor observations
        """
        pass
