"""Abstract base class for a Robot object."""

from abc import ABC, abstractmethod


class Robot(ABC):
    """Robot base class."""
    @abstractmethod
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def step(self, t, duration):
        """Take a step of a specified duration in the simulation.

        Returns any observations gathered during this step.

        Args:
            t (float): global time, in seconds
            duration (float): duration of step, in seconds.

        Returns (float, float): the last reported coordinate and
            observation. Returns None if no observations in com_window.
        """
        pass
