""" Defines a simulated robot that employs an offline planner."""
import operator

from .robot import Robot


class OfflineRobot(Robot):
    '''A point robot with a pre-programmed Trajectory and an Environment.'''

    def __init__(self, model, trajectory, environment, nominal_velocity,
                 com_window):
        """ Initialize Offline robot.

        Args:
            model (Model): a Model object to update
            trajectory (Trajectory): an offline Trajectory object to execute
            environment (Environment): an environment object
            nominal_velocity (float): velocity of robot in m/s
            com_window (float): communicate every com_window seconds
        """
        self.model = model
        self.trajectory = trajectory
        self.environment = environment
        self.vel = nominal_velocity
        self.com_window = com_window
        self.observation_buffer = {}
        self.current_observation = None

        self.last_report = 0.0  # last time a communication occurred
        self.current_distance = 0.0  # current distance along path

    def step(self, t, duration):
        """Take a step of a specified duration in the simulation.

        Returns any observations gathered during this step.

        Args:
            t (float): global time and the end of the step, in seconds
            duration (float): duration of step, in seconds.

        Returns (float, float): the last reported coordinate and
            observation. Returns None if no observations in com_window.
        """
        self.current_distance = self.vel * t
        # print(self.current_distance)

        # Get only returned x, y,(z) coordinate from trajectory
        self.coordinate = self.trajectory.path_sample(self.current_distance)[1:]

        samp = self.environment.get_value(t, self.coordinate)
        self.current_observation = samp
        self.observation_buffer[self.coordinate] = samp

        if t - self.last_report > self.com_window:
            report_coordinate = max(
                self.observation_buffer.items(), key=operator.itemgetter(1))[0]
            report_observation = max(
                self.observation_buffer.items(), key=operator.itemgetter(1))[1]
            self.observation_buffer = {}
            self.last_report = t
            return report_coordinate, report_observation
        else:
            return None

    def _json_stats(self):
        """Returns a dict of metadata about this robot."""
        json_config = {"nominal_velocity": self.vel,
                       "com_window": self.com_window}
        return json_config
