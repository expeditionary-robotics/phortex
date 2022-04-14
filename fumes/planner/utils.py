"""Utility objects for planning."""
import numpy as np

from scipy.optimize import NonlinearConstraint, LinearConstraint

# If True, the optimization is constrained to keep the solution feasbile
# at each iteration. In practice, this seems to really hurt optimizer
# performance. Recommended to set to False.
KEEP_FEASIBLE = False
THRESH_BUDGET_LB = 0.80


def soft_origin_penalty(theta, origin):
    """A soft penalty for distance of origin in parameter set from a
    soft origin.

    Args:
        theta (np.array): The input trajectory parameters, which contains
            the origin coordinates.
            TODO: could have the origin parameters passed in here, instead
            of the full parameters array
        origin (np.array): An array of length 2, encoding the end point
            of the previous trajectory.
    """
    # TODO: should make this parameter indexing more generalizable
    # Hard code, for our two paramter sets, which parameters correspond
    # to origin_x and origin_y.
    if len(theta) == 5:
        origin_x = theta[3]
        origin_y = theta[4]
    elif len(theta) == 3:
        origin_x = theta[1]
        origin_y = theta[2]
    else:
        raise ValueError("Unknown origin in parameter set.")
    return (origin_x - origin[0])**2 + (origin_y - origin[1])**2


def length_constraint(traj_generator, budget, method="SLSQP"):
    """Generates length constraint for optimization methods.

    For a given trajectory generator, generates a nonlinear
    constraint that enforces 1 <= trajectory length <= budget.

    Args:
        traj_generator (TrajectoryGenerator): input generator
        budget (float): length budget
        method  (str):  optimization method to generate constraints for.
    """

    def length_lb(theta):
        """Trajectory length >= 1"""
        return traj_generator(*theta).length - THRESH_BUDGET_LB*budget

    def length_ub(theta):
        """Trajectory length <= budget"""
        return -traj_generator(*theta).length + budget

    if method == "SLSQP":
        return [
            {"type": "ineq", "fun": length_lb},
            {"type": "ineq", "fun": length_ub},
        ]
        # return [
        #     {"type": "eq", "fun": length_ub},
        # ]
    elif method == "trust-constr":
        return [
            NonlinearConstraint(
                fun=lambda theta: traj_generator(*theta).length,
                jac="2-point",
                # lb=1.,
                lb=THRESH_BUDGET_LB*budget,
                ub=budget,
                keep_feasible=KEEP_FEASIBLE
            )
        ]


def bound_constraint(traj_generator, limits, method="SLSQP"):
    """Generates a boundary constraint for optimization methods.

    For a given trajectory generator, generates a nonlinear
    constraint that enforces that the trajectory remains inside of:
        limits[0] <= x <= limits[1].
        limits[2] <= y <= limits[3].

    Args:
        traj_generator (TrajectoryGenerator): input generator
        limits (tuple[float]): boundary in x and y, (xmin, xmax, ymin, ymax)
        method  (str):  optimization method to generate constraints for.
    """

    def x_lb(theta):
        """Trajectory xmin >= xmin"""
        return traj_generator(*theta).xmin - limits[0]

    def x_ub(theta):
        """Trajectory xmax <= xmax"""
        return -traj_generator(*theta).xmax + limits[1]

    def y_lb(theta):
        """Trajectory ymin >= ymin"""
        return traj_generator(*theta).ymin - limits[2]

    def y_ub(theta):
        """Trajectory ymax <= ymax"""
        return -traj_generator(*theta).ymax + limits[3]

    if method == "SLSQP":
        return [
            {"type": "ineq", "fun": x_lb},
            {"type": "ineq", "fun": x_ub},
            {"type": "ineq", "fun": y_lb},
            {"type": "ineq", "fun": y_ub},
        ]
    elif method == "trust-constr":
        return [
            NonlinearConstraint(
                fun=lambda theta: traj_generator(*theta).xmin,
                jac="2-point",
                lb=limits[0],
                ub=np.inf,
                keep_feasible=KEEP_FEASIBLE
            ),
            NonlinearConstraint(
                fun=lambda theta: traj_generator(*theta).xmax,
                jac="2-point",
                lb=-np.inf,
                ub=limits[1],
                keep_feasible=KEEP_FEASIBLE
            ),
            NonlinearConstraint(
                fun=lambda theta: traj_generator(*theta).ymin,
                jac="2-point",
                lb=limits[2],
                ub=np.inf,
                keep_feasible=KEEP_FEASIBLE
            ),
            NonlinearConstraint(
                fun=lambda theta: traj_generator(*theta).ymax,
                jac="2-point",
                lb=-np.inf,
                ub=limits[3],
                keep_feasible=KEEP_FEASIBLE
            ),
        ]


def param_constraint(param_bounds, method="SLSQP"):
    """Generates a bound constratins for all paramters.

    For a given trajectory generator, generates a linear
    constraint that enforces that a parameter remains inside of:
        param_bounds[i][0] <= param_i <= param_bounds[i][1]

    Args:
        param_bounds (tuple[float]): upper and lower limits for each parameter
        method  (str):  optimization method to generate constraints for.
    """
    if method == "SLSQP":
        c1 = [{"type": "ineq", "fun": lambda theta: -theta[i] + b[1]}
              for i, b in enumerate(param_bounds)]
        c2 = [{"type": "ineq", "fun": lambda theta: theta[i] - b[0]}
              for i, b in enumerate(param_bounds)]
        return c1 + c2
    elif method == "trust-constr":
        return [
            LinearConstraint(
                A=np.eye(len(param_bounds)),
                lb=np.array([b[0] for b in param_bounds]),
                ub=np.array([b[1] for b in param_bounds]),
                keep_feasible=[True] * len(param_bounds)
            )
        ]
