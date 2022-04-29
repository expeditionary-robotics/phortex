"""Utility file for saving and loading a mission."""
import os
from datetime import datetime
import pickle
import json


def get_mission_hash(mission_name="modelsim"):
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y-%H:%M:%S")
    return f"{mission_name}_{dt_string}"


def save_mission(mission_hash, rob, model, env=None, simulator=None):
    dirpath = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations")
    os.makedirs(dirpath, exist_ok=True)
    filepath = os.path.join(dirpath, f"{mission_hash}.pkl")
    with open(filepath, "wb") as fh:
        pickle.dump({
            "robot": rob,
            "model": model,
            "environment": env,
            "simulator": simulator,
        }, fh)


def load_mission(mission_hash):
    filepath = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/{mission_hash}.pkl")
    with open(filepath, "rb") as fh:
        data = pickle.load(fh)
        return data["robot"], data["model"], data["environment"], data["simulator"]


def save_experiment_json(experiment_name, iter_num, rob, model, env, traj_opt, trajectory, reward, simulation, experiment_dict):
    """Takes any definable experimental element and saves to JSON."""
    json_env_dict = env._json_stats()
    json_rob_dict = rob._json_stats()
    json_mod_dict = model._json_stats()
    json_traj_dict = trajectory._json_stats()
    json_reward_dict = reward._json_stats()
    json_sim_dict = simulation._json_stats()
    json_traj_opt_dict = traj_opt._json_stats()

    json_config_dict = {"environment_params": json_env_dict,
                        "model_params": json_mod_dict,
                        "robot_params": json_rob_dict,
                        "traj_params": json_traj_dict,
                        "traj_opt_params": json_traj_opt_dict,
                        "reward_params": json_reward_dict,
                        "simulation_params": json_sim_dict,
                        "experiment_params": experiment_dict
                        }

    directory = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/{experiment_name}")
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"exp_iter_{iter_num}.json")
    j_fp = open(filepath, 'w')
    json.dump(json_config_dict, j_fp)
    j_fp.close()


def load_experiment_json(experiment_name, iter_num):
    """Reads in any experiment JSON file."""
    filepath = os.path.join(os.getenv("FUMES_OUTPUT"),
                            f"simulations/{experiment_name}/exp_iter_{iter_num}.json")
    f = open(filepath)
    data = json.load(f)
    return data["environment_params"], data["model_params"], \
        data["robot_params"], data["traj_params"], data["traj_opt_params"], \
        data["reward_params"], data["experiment_params"]
