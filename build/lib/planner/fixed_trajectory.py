"""Fixed trajectory planner."""
from .planner import Planner


class FixedTrajectory(Planner):
    """Fixed trajectory planner."""

    def __init__(self, trajectory):
        """Initialize trajectory parameters.

        Args:
            trajectory (Trajectory): a Trajectory object
        """
        self.trajectory = trajectory

    def get_plan(self):
        """Return parameterized plan as a Trajectory object."""
        return self.trajectory
