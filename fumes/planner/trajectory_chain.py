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

            # # Get maximum value of environment at time t
            # # TODO: change this try/except interface to allow
            # # models to know if they're 3d or 2d
            try:
                # Try 3D get_maxima function
                # import pdb; pdb.set_trace()
                xm, ym, zm = self.planners[0].env_model.get_maxima(
                    t, z=[planner.traj_generator.alt])
            except Exception as e:
                # Except to 2D get_maxima function
                xm, ym = self.planners[0].env_model.get_maxima(t)
            x0 = list(planner.x0)
            try:
                x0[planner.param_names["origin_x"]] = xm
                x0[planner.param_names["origin_y"]] = ym
            except:
                import pdb; pdb.set_trace()
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
        # alt = [t.altitude for t in traj_chain]
        alt = traj_chain[0].altitude
        # [TODO]: we want chains to be able to have multiple altitudes
        return Chain(t0, vel, traj_chain, alt)
