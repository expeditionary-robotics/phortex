import numpy as np
from scipy.optimize import minimize
from scipy import optimize

from fumes.trajectory import Chain
from fumes.utils import tic, toc

from .planner import Planner

import pdb


class TrajectoryChain(Planner):
    def __init__(self, planners):
        """ Initialize trajectory optmizer.

        Args:
            planners (list[TrajectoryOpt]): list of trajectory optimizers
        """
        self.planners = planners

    def get_plan(self, from_cache=False):
        """Run the planner and return a Trajectory object.

        Args:
            from_cache (bool): if True, uses the model cache
                during planning.
        """
        traj_chain = []
        soft_origin = None  # origin constraint (should be near soft_origin)
        soft_com = None  # diversity constraint (should be far from soft_com)
        start_point = None  # start point for the trajectory
        t = self.planners[0].traj_generator.t0  # start time for the trajectory

        for i, planner in enumerate(self.planners):
            print(f"Planning chain {i} of {len(self.planners)} at time {t}")
            tic()
            # Override the start time and start point of each lawnmower
            planner.traj_generator.t0 = t  # TODO: should put this back in
            planner.traj_generator.start_point = start_point
            planner.id = f"chain{i}_"  # set the planner id

            # # Get maximum value of environment at time t
            # # TODO: change this try/except interface to allow
            # # models to know if they're 3d or 2d
            try:
                # Try 3D get_maxima function
                xm, ym, zm = self.planners[0].env_model.get_maxima(
                    t, z=[planner.traj_generator.alt])
                thm = self.planners[0].env_model.curr_head_sampler.heading(t) * 180. / np.pi
            except Exception as e:
                # Except to 2D get_maxima function
                xm, ym = self.planners[0].env_model.get_maxima(t)
                thm = self.planners[0].env_model.curr_head_sampler.heading(t) * 180. / np.pi

            # Set the initial parameter values based on the environment maximum
            x0 = list(planner.x0)
            x0[planner.param_names["origin_x"]] = xm
            x0[planner.param_names["origin_y"]] = ym

            if self.planners[i].traj_generator.start_point is not None: 
                # Ensure the travel distance + lawnmower size <= budget
                xs = self.planners[i].traj_generator.start_point[0]
                ys = self.planners[i].traj_generator.start_point[1]
                xo = self.planners[i].x0[planner.param_names["origin_x"]]
                yo = self.planners[i].x0[planner.param_names["origin_y"]]
                dist = np.sqrt((xs - xo)**2 + (ys - yo)**2)

                # Remaining budget after traveling to the origin 
                rem = self.planners[i].budget - dist

                # Lawnmower resolution 
                res = self.planners[i].traj_generator.res

                def lawn_length(lw, lh, res):
                    """Length of a lawnmower of a given lw, lw, and res"""
                    return lw * (lh // res + 1.0) + lh

                # Length of the current lawnmower 
                lw = self.planners[i].x0[planner.param_names["lw"]] 
                lh = self.planners[i].x0[planner.param_names["lh"]] 
                cur_len = lawn_length(lw, lh, res)

                def lawn_under_budget(budget, res):
                    """ Compute the largest squre lanwmower that satisfies the budget, 
                    assuming that lw = lh and res is fixed.

                    Solves:
                        budget = lw * (lh // res + 1.0) + lh
                        budget = lw * (lw // res + 1.0) + lw

                        budget = lw**2 // res + lw + lw
                        0 = lw**2 // res + 2 * lw  - budget

                        Solve quadratic equation .
                    """ 
                    lw = (-2.0 + np.sqrt(4.0 + (4.0 * budget) / res)) / (2.0 / res)
                    return lw

                if cur_len > rem: 
                    lw = lawn_under_budget(rem, res)
                    x0[planner.param_names["lw"]] = lw
                    x0[planner.param_names["lh"]] = lh

            # x0[planner.param_names["rot"]] = thm
            planner.x0 = tuple(x0)

            # Get the next trajectory in the chain
            traj_chain.append(planner.get_plan(soft_origin=soft_origin,
                                               soft_com=soft_com,
                                               from_cache=from_cache))

            print("\tPlanned parameters (lh, lw, orientation, origin_x, origin_y):",
                  traj_chain[-1].lh,
                  traj_chain[-1].lw,
                  traj_chain[-1].orientation,
                  traj_chain[-1].origin[0],
                  traj_chain[-1].origin[1])

            # Update the soft origin
            soft_origin = (
                traj_chain[-1].path.xy[0][-1], traj_chain[-1].path.xy[1][-1])

            # Get the center point of the previous trajectory
            soft_com = np.array(traj_chain[-1].path.xy).mean(axis=1)

            # Get the time for the next trajectory to start optimizing from
            t = traj_chain[-1].time_at_end

            # Get the origin for the next trajectory to start from
            start_point = soft_origin

            toc()

        t0 = traj_chain[0].t0
        vel = traj_chain[0].vel
        # [TODO]: we want chains to be able to have multiple altitudes
        # alt = [t.altitude for t in traj_chain]
        alt = traj_chain[0].altitude
        return Chain(t0, vel, traj_chain, alt)
