"""Iterates through paper results and computes metrics"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from fumes.environment.mtt import CrossflowMTT
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile

from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, curfunc, headfunc
from fumes.utils.save_mission import load_experiment_json

SAVE_DIRECTORY = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/summary")
if not os.path.exists(SAVE_DIRECTORY):
        os.makedirs(SAVE_DIRECTORY)
EXP_PREFIX = "cloud_papermtt_iterativeplans_"
EXPS_TO_PROCESS = ["seed102", "seed851", "seed42", "seed87", "seed281", "seed647", "seed653"]
ITER_NUMS = [0, 1, 2, 3, 4]
EXTENT = Extent(xrange=(-100., 500.),
                xres=100,
                yrange=(-100., 500.),
                yres=100,
                zrange=(0, 200),
                zres=50,
                global_origin=(0., 0., 0.))

if __name__ == "__main__":
    results = {"exp_prefix": EXP_PREFIX,
               "exp_seeds": EXPS_TO_PROCESS,
               "iters": ITER_NUMS}

    for iter in ITER_NUMS:
        # initialize storage
        results[f"rmse_{iter}"] = []
        results[f"class_error_{iter}"] = []
        results[f"total_detects_{iter}"] = []
        results[f"prop_detects_{iter}"] = []
        results[f"spatial_extent_{iter}"] = []
        results[f"temporal_util_{iter}"] = []

    # set the queries for computing error
    queryx, queryy = np.meshgrid(np.linspace(-100, 500, 25), np.linspace(-100, 500, 25))

    # get the environment (since the same, just need to do this once)
    env_params = load_experiment_json(
        EXP_PREFIX + EXPS_TO_PROCESS[0], ITER_NUMS[0])["environment_params"]
    env = CrossflowMTT(extent=EXTENT,
                       plume_loc=env_params["plume_loc"],
                       s=env_params["s"],
                       curfunc=curfunc,
                       headfunc=headfunc,
                       tprof=pacific_sp_T,
                       sprof=pacific_sp_S,
                       rhoprof=eos_rho,
                       vex=env_params["vex"],
                       area=env_params["area"],
                       density=env_params["density"],
                       salt=env_params["salt"],
                       temp=env_params["temp"],
                       lam=env_params["lam"],
                       entrainment=env_params["entrainment"])

    # Read in each simulation json
    for exp in EXPS_TO_PROCESS:
        for iter in ITER_NUMS:
            exp_json = load_experiment_json(EXP_PREFIX + exp, iter)
            times = np.asarray(exp_json["simulation_params"]["times"])
            snap_times = np.unique(np.round(times / 3600.))

            # get the trained model
            mod_params = exp_json["model_params"]
            mod = CrossflowMTT(extent=EXTENT,
                               plume_loc=mod_params["model_fixed_params"]["plume_loc"],
                               s=mod_params["model_fixed_params"]["s"],
                               curfunc=curfunc,
                               headfunc=headfunc,
                               tprof=pacific_sp_T,
                               sprof=pacific_sp_S,
                               rhoprof=eos_rho,
                               vex=mod_params["model_learned_params"]["velocity_mle"],
                               area=mod_params["model_learned_params"]["area_mle"],
                               density=mod_params["model_fixed_params"]["density"],
                               salt=mod_params["model_fixed_params"]["salt"],
                               temp=mod_params["model_fixed_params"]["temp"],
                               lam=env_params["lam"],
                               entrainment=(mod_params["model_learned_params"]["entrainment_alpha_mle"], mod_params["model_learned_params"]["entrainment_beta_mle"]))

            # compute the RMSE between each simulated hour snapshot and the true underlying environment
            # for a random sample of points
            all_env_obs = []
            all_mod_obs = []
            for t in snap_times:
                env_samps = env.get_value(
                    t=t * 3600., loc=(queryx.flatten(), queryy.flatten(), np.ones_like(queryx.flatten()) * 80.))
                all_env_obs.append(env_samps > 1e-5)
                mod_samps = mod.get_value(
                    t=t * 3600., loc=(queryx.flatten(), queryy.flatten(), np.ones_like(queryx.flatten()) * 80.))
                all_mod_obs.append(mod_samps > 1e-5)
            rmse = np.sqrt(np.mean((np.asarray(all_env_obs).flatten().astype(
                float) - np.asarray(all_mod_obs).flatten().astype(float))**2))
            class_error = np.sum(np.fabs(np.asarray(all_env_obs).flatten().astype(
                float) - np.asarray(all_mod_obs).flatten().astype(float))) / len(np.asarray(all_env_obs).flatten())
            results[f"rmse_{iter}"].append(rmse)
            results[f"class_error_{iter}"].append(class_error)
            print(f"RMSE: {rmse}")
            print(f"Classificaton Error: {class_error}")

            # grab total samples and proportion samples for each simulation
            total_detects = exp_json["experiment_params"]["total_in_plume_samples"]
            prop_detects = exp_json["experiment_params"]["portion_in_plume_samples"]
            results[f"total_detects_{iter}"].append(total_detects)
            results[f"prop_detects_{iter}"].append(prop_detects)
            print(f"Total Detections: {total_detects}")
            print(f"Total Prop Detections: {prop_detects}")

            # get simulated observations
            sim_obs = np.asarray(exp_json["simulation_params"]["obs"])
            sim_coords = exp_json["simulation_params"]["coords"]

            # grab spatial extent (furthest detecton over furthest distance)
            distances = np.asarray([np.sqrt((coord[0])**2 + (coord[1])**2) for coord in sim_coords])
            max_dist = np.nanmax(distances)
            max_detect_dist = np.nanmax(distances[sim_obs > 1e-5])
            print(max_detect_dist, max_dist)
            spatial_extent = max_detect_dist / max_dist
            results[f"spatial_extent_{iter}"].append(spatial_extent)
            print(f"Spatial Extent: {spatial_extent}")

            # grab temporal utilization (number hours with 10% or more detections over total hours)
            h_count = 0
            for h in snap_times:
                idt = times < (h * 3600.)
                prop = np.sum((sim_obs[idt] > 1e-5)) / len(sim_obs[idt])
                if prop >= 0.1:
                    h_count += 1
            temporal_util = h_count / len(snap_times)
            results[f"temporal_util_{iter}"].append(temporal_util)
            print(f"Temporal Utilization: {temporal_util}")
            print("-----")

    # plot "pretty" composite graphs of metrics
    plt.violinplot([results[f"rmse_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.xlabel("Iteration Number")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"rmse.svg"))
    plt.close()

    plt.violinplot([results[f"class_error_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.xlabel("Iteration Number")
    plt.ylabel("Classification Error")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"class_error.svg"))
    plt.close()

    plt.violinplot([results[f"total_detects_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[8735], xmin=[0], xmax=[4], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Total Positive Detections")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"total_detects.svg"))
    plt.close()

    plt.violinplot([results[f"prop_detects_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[0.202], xmin=[0], xmax=[4], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Proportion Positive Detections")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"prop_detects.svg"))
    plt.close()

    plt.violinplot([results[f"spatial_extent_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.xlabel("Iteration Number")
    plt.ylabel("Spatial Utilization")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"spatial.svg"))
    plt.close()

    plt.violinplot([results[f"temporal_util_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.xlabel("Iteration Number")
    plt.ylabel("Temporal Utilization")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"temporal.svg"))
    plt.close()

    # Save the results
    filepath = os.path.join(SAVE_DIRECTORY, f"results.json")
    j_fp = open(filepath, 'w')
    json.dump(results, j_fp)
    j_fp.close()
