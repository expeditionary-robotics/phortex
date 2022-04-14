"""Standard interface for Model objects."""
# TODO: what kind of interface do we want our model objects to have?

from abc import ABC, abstractmethod


class Model(ABC):
    """Abstract Model base class."""

    def __init__(self, extent, **kwargs):
        """Initialize model.

        Args:
            extent (Extent): a geographic Extent object
        """
        pass

    @abstractmethod
    def get_maxima(self, t):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time

        Returns: tuple, maximum location in x, y, z
        """
        pass

    @abstractmethod
    def get_value(self, t, loc):
        """Get a deterministic prediction of the model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz or xy space

        Returns: (???) TODO: some kind of standard return?
        """
        pass

    @abstractmethod
    def get_prediction(self, t, loc):
        """Get a (mean, variance) prediction of the model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz or xy space

        Returns: tuple(???): some kind of standard return with
            mean and variance?
        """
        pass

    @abstractmethod
    def get_snapshot(self, t):
        """Get a deterministic prediction of full state at time t.

        Args:
            t (float): the global time

        Returns: (???) TODO: some kind of standard snapshot return?
        """
        pass

    @abstractmethod
    def get_snapshot_prediction(self, t):
        """Get a deterministic prediction of full state and uncertainty at t.

        Args:
            t (float): the global time

        Returns: (???) TODO: some kind of standard tuple of snapshot return?
            and uncertainty
        """
        pass

    @abstractmethod
    def update(self, t, loc, obs):
        """Update model with observation at t, loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz or xy space
            obs (tuple[float]): a sensor observation
        """
        pass


class ScienceModel(Model):
    """Abstract ScienceModel base class."""

    def __init__(self, extent, **kwargs):
        """Initialize science model.

        Args:
            extent (Extent): a geographic Extent object
        """
        pass

    @abstractmethod
    def get_parameters(self):
        """Returns the parameters defining the model."""
        pass

    @abstractmethod
    def solve(self):
        """Given new parameter settings, compute a new science model."""
        pass
