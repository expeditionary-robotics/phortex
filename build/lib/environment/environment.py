"""Standard interface for Environment objects."""
# TODO: what kind of interface do we want our environment objects to have?

from abc import ABC, abstractmethod


class Environment(ABC):
    """Abstract Environment base class."""

    def __init__(self, extent, **kwargs):
        """Initialize enviornment."""
        pass

    @abstractmethod
    def get_value(self, t, loc):
        """Get ground truth at time t and location loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz or xy space

        Returns: the concentration value
        """
        pass

    @abstractmethod
    def get_snapshot(self, t):
        """Get a ground truth full state at time t.

        Args:
            t (float): the global time

        Returns: (???) TODO: some kind of standard snapshot return?
        """
        pass


class BackgroundModel(ABC):
    """Abstract Background base class."""

    def __init__(self, **kwargs):
        """Initialize the background profile."""
        pass

    @abstractmethod
    def define_from_data(self, data):
        """Perform linear fitting on a given dataset.

        Args:
            data (pandas dataframe): data to fit linear curve

        Returns:
            slope (float) and intercept (float) of data
        """
        pass

    @abstractmethod
    def profile(self):
        """Returns functional for use in model class."""
        pass

    @abstractmethod
    def get_slope(self):
        """Returns slope value."""
        pass

    @abstractmethod
    def get_intercept(self):
        """Returns intercept value."""
        pass
