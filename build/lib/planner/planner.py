from abc import ABC, abstractmethod


class Planner(ABC):
    def __init__(self, **kwargs):
        """Abstract Planner base class."""
        pass

    @abstractmethod
    def get_plan(self, from_cache=False):
        """Run the planner and return a Trajectory object.

        Args:
            from_cache (bool): if True, uses the model cache
                during planning.
        """
        pass
