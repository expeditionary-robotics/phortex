import numpy as np
from scipy import optimize
from scipy.optimize import minimize

import os

import matplotlib.pyplot as plt

from fumes.utils import tic, toc
from fumes.simulator.utils import visualize_and_save_traj

from .planner import Planner
from .utils import length_constraint, bound_constraint, param_constraint, \
    soft_origin_penalty


class TrajectoryOpt(Planner):
    def __init__(self, env_model, traj_generator, reward, x0, budget=None,
                 limits=[0., 1000., 0., 1000.], param_bounds=None,
                 param_names=None, max_iters=30, tol=1e-8,
                 method="trust-constr", experiment_name=None):
        """ Initialize trajectory optmizer.

        Args:
            env_model (Model/Environment): a Model or Environemnt object,
                supporting method `get_val`
            traj_generator (TrajectoryGenerator): a TrajectoryGenerator object
            reward (Reward): a Reward object that takes a Trajectory as input
                and outputs a reward score
            x0 (np.array): input parameter values that can be fed to the
                traj_generator object to produce a trajectory, e.g.,
                np.array([height, width, theta, resolution])
            budget (float): if a length budget is provided, the returned
                trajectory will be constrained
            limits (tuple[float]): axis-aligned safety zone, in x and y,
                e.g., (xmin, xmax, ymin, ymax)
            param_bounds (list[tuple]): list of upper and lower bounds
                for parameters
            param_names (dict): dictionary from parameter names to index
                in parameter vector
            max_iters (int): maximum number of optimization iterations
            tol (float): algorithm convergence tolerance
            method (str): optimization method, one of:
                "SLSQP", "trust-constr", "BFGS", "basinhopping"
            experiment_name (str): the name of the experiment (optional), used
                to name output files.
        """
        self.env_model = env_model
        self.traj_generator = traj_generator
        self.reward = reward
        self.x0 = x0
        self.budget = budget
        self.limits = limits
        self.method = method
        self.param_bounds = param_bounds
        self.param_names = param_names
        self.max_iters = max_iters
        self.tol = tol
        self.experiment_name = experiment_name
        self.reward_history = []

        if self.experiment_name is None:
            self.experiment_name = "temp"
        self.id = ""

        self.neval = 1

        # Generate planning path, if needed
        self.path = os.path.join(os.getenv("FUMES_OUTPUT"), "planning", self.experiment_name)
        os.makedirs(self.path, exist_ok=True)

    def _json_stats(self):
        """Returns a dict of info about this optimizer."""
        json_dict = {"x0": self.x0,
                     "budget": self.budget,
                     "limits": self.limits,
                     "method": self.method,
                     "param_bounds": self.param_bounds,
                     "max_iters": self.max_iters,
                     "tol": self.tol}
        return json_dict

    def get_plan(self, soft_origin=None, soft_com=None, from_cache=False):
        """Get a plan by minimizing a cost funciton.

        Args:
            soft_origin (tuple[float]): if not None, origin is encouraged to
                be near soft_origin with a soft constraint
            soft_com(tuple[float]): if not None, samples are encouraged to
                be far from soft_com with a soft constraint
            from_cache (bool): if True, uses the model cache
                during planning.
        """
        ###########################
        #### Setup constraints ####
        ###########################
        con = []
        if self.budget is not None:
            print("Adding budget constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            con += length_constraint(
                self.traj_generator.generate, budget=self.budget, method=self.method)

        if self.limits is not None:
            print("Adding safety boundary constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            con += bound_constraint(
                self.traj_generator.generate, limits=self.limits, method=self.method)

        if self.param_bounds is not None:
            print("Adding parameter bounds constraint.")
            if self.method != "SLSQP" and self.method != "trust-constr":
                raise ValueError("Cannot perform constrained optimization.")

            # Instantiate constraint
            con += param_constraint(param_bounds=self.param_bounds, method=self.method)

        ##############################
        #### Add soft constraints ####
        ##############################
        def rew(theta):
            return self.reward.eval(
                self.traj_generator.generate(*theta),
                self.env_model,
                from_cache=from_cache)

        if soft_origin is not None:
            print("Added soft origin constraint.")

        def s_origin(theta):
            if soft_origin is not None:
                # return soft_origin_penalty((theta[self.param_names['origin_x']],
                #                             theta[self.param_names['origin_y']]),
                #                            soft_origin)
                # [TODO] do we want a soft origin?
                return 0.0
            else:
                return 0.0

        if soft_com is not None:
            print("Added a diversity incentive.")

        def s_com(theta):
            if soft_com is not None:
                # return soft_origin_penalty((theta[self.param_names['origin_x']],
                #                             theta[self.param_names['origin_y']]),
                #                            soft_com)
                # [TODO] do we want a diversity bonus?
                return 0.0
            else:
                return 0.0

        def fun(theta):
            return rew(theta) + s_origin(theta) + s_com(theta)

        ############################
        #### Solve optimization ####
        ############################

        # print(">>>>>Generator time.")
        # tic()
        # traj = self.traj_generator.generate(*self.x0)
        # toc()
        # tic()
        # test = self.reward.eval(traj, self.env_model, from_cache=from_cache)
        # toc()
        # print("<<<<<Done.")

        if self.max_iters is None:
            options = {'disp': True}
        else:
            options = {'disp': True, 'maxiter': self.max_iters}

        if self.tol is not None:
            if self.method == "SLSQP":
                options['ftol'] = self.tol
            elif self.method == "trust-constr":
                options['gtol'] = self.tol
            elif self.method == "BFGS":
                options['xtol'] = self.tol

        if self.method == "SLSQP" or \
                self.method == "trust-constr" or \
                self.method == "BFGS":

            res = optimize.minimize(
                fun=fun,
                jac=None,
                x0=np.array(self.x0),
                options=options,
                constraints=con,
                method=self.method,
                callback=self._callback)
        elif self.method == "basinhopping":
            res = optimize.basinhopping(
                func=lambda theta: self.reward.eval(
                    self.traj_generator.generate(*theta), self.env_model),
                x0=np.array(self.x0),
                constraints=con,
                disp=True,
                callback=self._callback)
        else:
            raise ValueError(f"Unrecognized optimization method {self.method}."
                             f"Should be one of SLSQP, trust-constr, BFGS, or basinhopping.")

        print("Optimization completed. Result:", res.x)
        print("Length:", self.traj_generator.generate(*res.x).length)
        return self.traj_generator.generate(*res.x)

    def _callback(self, x, *args):
        """This callback is called during every iteration of optimization."""
        # rew = self.reward.eval(self.traj_generator.generate(*x), self.env_model)
        # import pdb; pdb.set_trace()
        rew = args[0].fun
        self.reward_history.append(rew)
        plt.plot(range(len(self.reward_history)), self.reward_history)
        plt.xlabel("Iterations")
        plt.ylabel("(Negative) Reward")
        plt.title("Optimization progress (should go down)")
        plt.savefig(os.path.join(self.path, f"training_progress_{self.id}plot.png"))
        plt.close()

        print(
            # f"n:{self.neval}\t Value:{self.reward.eval(self.traj_generator.generate(*x), self.env_model)}")
            f"n:{self.neval}\t Value:{rew}")
            # f"n:{self.neval}")
        if not (self.neval % 10):
            print(
                f"\t n:{self.neval}\t Value:{self.reward.eval(self.traj_generator.generate(*x), self.env_model)}")
            print("Saving mission checkpoint.")

            traj = self.traj_generator.generate(*x)
            visualize_and_save_traj(
                traj,
                self.env_model.extent,
                traj_name=os.path.join(self.path, f"temp_{self.id}iter{self.neval}"))
            print("Done.")
        self.neval += 1
