"""Plot a saved simulation by mission hash.

Example usage:
    python fumes/examples/post_cruise_examples/demo3-plot-from-saved-sim.py -m crossflow_gp_cache_14112021-15:08:40
"""
import os
import argparse
import numpy as np
from fumes.utils import tic, toc
from fumes.utils.save_mission import load_experiment_json, \
    print_experiment_json_summary

# Experiment to examine
experiment_name = "stationarymtt_iterativeplans"
iter_num = 0  # which output to look at

# Get the data
json_dict = load_experiment_json(experiment_name, iter_num)
# print_experiment_json_summary(json_dict)

# Seperate out the components
env_dict = json_dict["environment_params"]
model_dict = json_dict["model_params"]
robot_dict = json_dict["robot_params"]
traj_dict = json_dict["traj_params"]
traj_opt_dict = json_dict["traj_opt_params"]
reward_dict = json_dict["reward_params"]
sim_dict = json_dict["simulation_params"]
exp_dict = json_dict["experiment_params"]

# Answer simple queries from JSON
print_dict = {"num_obs": exp_dict["total_samples"],  # total obs
              "num_in_plume_obs": exp_dict["total_in_plume_samples"],  # obs in plume
              "model_V": model_dict["model_learned_params"]["velocity_mle"],  # learned V
              "model_A": model_dict["model_learned_params"]["area_mle"],  # learned A
              "model_E": model_dict["model_learned_params"]["entrainment_mle"],  # learned E
              "env_V": env_dict["velocity"],  # true V
              "env_A": env_dict["area"],  # true A
              "env_E": env_dict["entrainment"],  # true E
              }
print_experiment_json_summary(print_dict)

# Answer complex queries from JSON
# Example: Get the observations from locations further than
# a certain distance from the vent
path_coords = sim_dict["coords"],  # trajectory coordinates
path_obs = np.asarray(sim_dict["obs"])  # trajectory observations
dist_query = 180.  # meters
vent_loc = env_dict["plume_loc"]  # vent location
coord_dists = np.asarray([np.sqrt((l[0] - vent_loc[0])**2 + (l[1] - vent_loc[1]) **
                       2 + (l[2] - vent_loc[2])**2) for l in path_coords[0]])
dist_mask = coord_dists > dist_query
further_obs = path_obs[dist_mask]
print(f"num obs over {dist_query}m away: {len(further_obs)}")
