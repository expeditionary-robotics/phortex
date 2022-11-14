"""Iterates through paper results and computes metrics"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from fumes.environment.mtt import CrossflowMTT
from fumes.environment.extent import Extent
from fumes.environment.profile import Profile

from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, curfunc, headfunc
from fumes.utils.save_mission import load_experiment_json

SAVE_DIRECTORY = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/summary_1112_100m")
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
EXP_PREFIX = "cloud_paperdive_iterativeplans_"
# EXP_PREFIX = "cloud_phortexphumes_iterativeplans_seed"
# EXPS_TO_PROCESS = ["seed55", "seed122", "seed217", "seed550", "seed593",
#                    "seed51", "seed464", "seed446", "seed847", "seed742"]
EXPS_TO_PROCESS = ["seed583", "seed458", "seed815", "seed319", "seed482", "seed597",
                   "seed625", "seed523", "seed243", "seed782"]
# EXPS_TO_PROCESS = ["37", "154", "225", "289", "479", "639", "685", "707", "709", "958"]
# EXPS_TO_PROCESS = ["20", "62", "125", "291", "139", "495", "592", "617", "301"]
# set the queries for computing error
queryz = [100., 150., 200.]
ITER_NUMS = [0, 1, 2]
EXTENT = Extent(xrange=(-100., 500.),
                xres=10,
                yrange=(-100., 500.),
                yres=10,
                zrange=(0, 200),
                zres=2,
                global_origin=(0., 0., 0.))
# alpha_prior: 0.15, beta_prior: 0.25, vel_prior: 1.5, area_prior: 0.5
# alpha_prior: 0.15, beta_prior: 0.15, vel_prior: 0.775, area_prior: 0.275
BASELINE = CrossflowMTT(extent=EXTENT,
                        plume_loc=(0., 0., 0.),
                        s=np.linspace(0, 500, 100),
                        curfunc=curfunc,
                        headfunc=headfunc,
                        tprof=pacific_sp_T,
                        sprof=pacific_sp_S,
                        rhoprof=eos_rho,
                        vex=0.775,
                        area=0.5,
                        density=eos_rho(300, 34.608),
                        salt=34.608,
                        temp=300,
                        lam=1.0,
                        entrainment=[0.16, 0.16])
area_mle = [0.5]
velocity_mle = [0.775]

if __name__ == "__main__":
    results = {"exp_prefix": EXP_PREFIX,
               "exp_seeds": EXPS_TO_PROCESS,
               "iters": ITER_NUMS}

    for iter in ITER_NUMS:
        # initialize storage
        results[f"rmse_{iter}"] = []
        results[f"class_error_{iter}"] = []
        results[f"iou_{iter}"] = []
        results[f"total_detects_{iter}"] = []
        results[f"prop_detects_{iter}"] = []
        results[f"spatial_extent_{iter}"] = []
        results[f"temporal_util_{iter}"] = []
        results[f"heat_flux_error_{iter}"] = []
        results[f"momentum_flux_error_{iter}"] = []
        results[f"buoyancy_flux_error_{iter}"] = []
        results[f"heat_flux_error_redux_{iter}"] = []
        results[f"momentum_flux_error_redux_{iter}"] = []
        results[f"buoyancy_flux_error_redux_{iter}"] = []
        results[f"area_mle_{iter}"] = []
        results[f"area_samples_{iter}"] = []
        results[f"area_error_{iter}"] = []
        results[f"velocity_mle_{iter}"] = []
        results[f"velocity_samples_{iter}"] = []
        results[f"velocity_error_{iter}"] = []
        results[f"alpha_error_{iter}"] = []
        results[f"beta_error_{iter}"] = []

    # get the environment (since the same, just need to do this once)
    exemplar_params = load_experiment_json(
        EXP_PREFIX + EXPS_TO_PROCESS[0], ITER_NUMS[0])
    env_params = exemplar_params["environment_params"]
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

    # compute baseline rmse, class_error
    times = np.asarray(exemplar_params["simulation_params"]["times"])
    snap_times = np.unique(np.round(times / 3600.))
    all_env_obs = []
    all_mod_obs = []
    for t in snap_times:
        env_samps = env.get_snapshot(t=t * 3600., z=queryz, from_cache=False)
        for samp in env_samps:
            all_env_obs.append(samp.flatten() > 1e-5)
        mod_samps = BASELINE.get_snapshot(t=t * 3600., z=queryz, from_cache=False)
        for samp in mod_samps:
            all_mod_obs.append(samp.flatten() > 1e-5)
    obs_in_both = (np.asarray(all_env_obs).flatten().astype(float) +
                   np.asarray(all_mod_obs).flatten().astype(float))
    intersection = len(obs_in_both[obs_in_both >= 2])
    union = len(obs_in_both[obs_in_both >= 1])
    rmse = np.sqrt(np.mean((np.asarray(all_env_obs).flatten().astype(
        float) - np.asarray(all_mod_obs).flatten().astype(float))**2))
    class_error = np.sum(np.fabs(np.asarray(all_env_obs).flatten().astype(
        float) - np.asarray(all_mod_obs).flatten().astype(float)))
    iou = float(intersection / union)
    results[f"rmse_init"] = rmse
    results[f"class_error_init"] = class_error
    results[f"iou_init"] = iou
    print("Initialized model...")
    print(f"RMSE: {rmse}")
    print(f"Classificaton Error: {class_error}")
    print(f"IOU: {iou}")

    # compute the flux error
    init_heatflux = (env.q0 - BASELINE.q0) / env.q0
    init_momflux = (env.m0 - BASELINE.m0) / env.m0
    init_buoyflux = (env.f0 - BASELINE.f0) / env.f0
    results["heat_flux_error_init"] = (env.q0 - BASELINE.q0) / env.q0
    results["momentum_flux_error_init"] = (env.m0 - BASELINE.m0) / env.m0
    results["buoyancy_flux_error_init"] = (env.f0 - BASELINE.f0) / env.f0
    print("Heat flux error ", results["heat_flux_error_init"])
    print("Momentum flux error ", results["momentum_flux_error_init"])
    print("Buoyancy flux error ", results["buoyancy_flux_error_init"])
    print("---")

    # compute error in the params of interest
    results["velocity_error_init"] = (env.v0 - BASELINE.v0) / env.v0
    results["area_error_init"] = (env.a0 - BASELINE.a0) / env.a0
    results["alpha_error_init"] = (
        env.entrainment[0] - BASELINE.entrainment[0]) / env.entrainment[0]
    results["beta_error_init"] = (env.entrainment[1] - BASELINE.entrainment[1]) / env.entrainment[1]

    # set up chain plots
    # fig, ax = plt.subplots(4, 1, sharex=True)
    # ax[0].hlines([env.a0], [0], [650], color="red")
    # ax[0].set_ylabel("Area")
    # ax[1].hlines([env.v0], [0], [650], color="red")
    # ax[1].set_ylabel("Velocity")
    # ax[2].hlines([env.entrainment[0]], [0], [650], color="red")
    # ax[2].set_ylabel("Alpha")
    # ax[3].hlines([env.entrainment[0]], [0], [650], color="red")
    # ax[3].set_ylabel("Beta")

    # all_area = []
    # all_velocity = []
    # all_alpha = []
    # all_beta = []

    # Read in each simulation json
    for exp in EXPS_TO_PROCESS:
        for iter in ITER_NUMS:
            print(exp, iter)
            exp_json = load_experiment_json(EXP_PREFIX + exp, iter)
            times = np.asarray(exp_json["simulation_params"]["times"])
            snap_times = np.unique(np.round(times / 3600.))

            # get the trained model
            mod_params = exp_json["model_params"]
            # chain = np.asarray(mod_params["model_update_procedure"]["chain_samples"])
            # all_area.append(chain[:, 3][150:])
            # all_velocity.append(chain[:, 2][150:])
            # all_alpha.append(chain[:, 0][150:])
            # all_beta.append(chain[:, 1][150:])
            # ax[0].plot(chain[:, 3][150:])
            # ax[1].plot(chain[:, 2][150:])
            # ax[2].plot(chain[:, 0][150:])
            # ax[3].plot(chain[:, 1][150:])
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
            results[f"area_mle_{iter}"].append(mod.a0)
            results[f"velocity_mle_{iter}"].append(mod.v0)
            # compute the RMSE between each simulated hour snapshot and the true underlying environment
            # for a random sample of points
            all_env_obs = []
            all_mod_obs = []
            for t in snap_times:
                env_samps = env.get_snapshot(t=t * 3600., z=queryz, from_cache=False)
                for samp in env_samps:
                    all_env_obs.append(samp.flatten() > 1e-5)
                mod_samps = mod.get_snapshot(t=t * 3600., z=queryz, from_cache=False)
                for samp in mod_samps:
                    all_mod_obs.append(samp.flatten() > 1e-5)
            obs_in_both = (np.asarray(all_env_obs).flatten().astype(float) +
                           np.asarray(all_mod_obs).flatten().astype(float))
            intersection = len(obs_in_both[obs_in_both >= 2.])
            union = len(obs_in_both[obs_in_both >= 1.])
            rmse = np.sqrt(np.mean((np.asarray(all_env_obs).flatten().astype(
                float) - np.asarray(all_mod_obs).flatten().astype(float))**2))
            class_error = np.sum(np.fabs(np.asarray(all_env_obs).flatten().astype(
                float) - np.asarray(all_mod_obs).flatten().astype(float)))
            iou = float(intersection / union)
            results[f"rmse_{iter}"].append(rmse)
            results[f"class_error_{iter}"].append(class_error)
            results[f"iou_{iter}"].append(iou)
            print(f"RMSE: {rmse}")
            print(f"Classificaton Error: {class_error}")
            print(f"IOU: {iou}")

            # get science error
            results[f"heat_flux_error_{iter}"] = (env.q0 - mod.q0) / env.q0
            results[f"momentum_flux_error_{iter}"] = (env.m0 - mod.m0) / env.m0
            results[f"buoyancy_flux_error_{iter}"] = (env.f0 - mod.f0) / env.f0
            hfer = (init_heatflux - results[f"heat_flux_error_{iter}"]) / init_heatflux
            mfer = (init_momflux - results[f"momentum_flux_error_{iter}"]) / init_momflux
            bfer = (init_buoyflux - results[f"buoyancy_flux_error_{iter}"]) / init_buoyflux
            results[f"heat_flux_error_redux_{iter}"] = hfer
            results[f"momentum_flux_error_redux_{iter}"] = mfer
            results[f"buoyancy_flux_error_redux_{iter}"] = bfer
            print("Heat flux error ", results[f"heat_flux_error_{iter}"])
            print("Momentum flux error ", results[f"momentum_flux_error_{iter}"])
            print("Buoyancy flux error ", results[f"buoyancy_flux_error_{iter}"])
            print("Heat flux error redux ", (init_heatflux -
                  results[f"heat_flux_error_{iter}"]) / init_heatflux)
            print("Momentum flux error redux ", (init_momflux -
                  results[f"momentum_flux_error_{iter}"]) / init_momflux)
            print("Buoyancy flux error redux ", (init_buoyflux -
                  results[f"buoyancy_flux_error_{iter}"]) / init_buoyflux)

            # get param error
            results[f"velocity_error_{iter}"].append((env.v0 - mod.v0) / env.v0)
            results[f"area_error_{iter}"].append((env.a0 - mod.a0) / env.a0)
            results[f"alpha_error_{iter}"].append(
                (env.entrainment[0] - mod.entrainment[0]) / env.entrainment[0])
            results[f"beta_error_{iter}"].append(
                (env.entrainment[1] - mod.entrainment[1]) / env.entrainment[1])

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
            spatial_extent = max_detect_dist / max_dist
            results[f"spatial_extent_{iter}"].append(spatial_extent)
            print(f"Spatial Extent: {spatial_extent}")

            # grab temporal utilization (number hours with 10% or more detections over total hours)
            h_count = 0
            lower = 0
            for h in snap_times:
                idt = (times >= lower) & (times <= (h * 3600.))
                lower = h * 3600.
                prop = np.sum((sim_obs[idt] > 1e-5)) / len(sim_obs[idt])
                if prop >= 0.1:
                    h_count += 1
            temporal_util = h_count / len(snap_times)
            results[f"temporal_util_{iter}"].append(temporal_util)
            print(f"Temporal Utilization: {temporal_util}")
            print("-----")

            # results[f"velocity_samples_{iter}"].append(np.asarray(
            #     mod_params["model_learned_params"]["velocity_samples"]).flatten())
            # results[f"area_samples_{iter}"].append(np.asarray(
            #     mod_params["model_learned_params"]["area_samples"]).flatten())

    # plt.show()

    # plt.hist(np.asarray(all_area).flatten(), bins=50, density=True)
    # plt.vlines([env.a0], 0, 10, color="red")
    # plt.show()

    # plt.hist(np.asarray(all_velocity).flatten(), bins=50, density=True)
    # plt.vlines([env.v0], 0, 10, color="red")
    # plt.show()

    # plt.hist(np.asarray(all_alpha).flatten(), bins=50, density=True)
    # plt.vlines([env.entrainment[0]], 0, 10, color="red")
    # plt.show()

    # plt.hist(np.asarray(all_beta).flatten(), bins=50, density=True)
    # plt.vlines([env.entrainment[1]], 0, 10, color="red")
    # plt.show()
    
    # plot "pretty" composite graphs of metrics
    plt.violinplot([results[f"rmse_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[results["rmse_init"]], xmin=[0], xmax=[ITER_NUMS[-1]], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"rmse.svg"))
    plt.close()

    plt.violinplot([results[f"class_error_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[results["class_error_init"]], xmin=[0], xmax=[ITER_NUMS[-1]], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Classification Error")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"class_error.svg"))
    plt.close()

    plt.violinplot([results[f"iou_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[results["iou_init"]], xmin=[0], xmax=[ITER_NUMS[-1]], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Intersection over Union")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"iou.svg"))
    plt.close()

    plt.violinplot([results[f"area_error_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[results["area_error_init"]], xmin=[0], xmax=[ITER_NUMS[-1]], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Error in Area Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"area.svg"))
    plt.close()

    plt.violinplot([results[f"velocity_error_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[results["velocity_error_init"]], xmin=[0], xmax=[ITER_NUMS[-1]], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Error in Velocity Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"velocity.svg"))
    plt.close()

    plt.violinplot([results[f"alpha_error_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[results["alpha_error_init"]], xmin=[0], xmax=[ITER_NUMS[-1]], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Error in Alpha Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"alpha.svg"))
    plt.close()

    plt.violinplot([results[f"beta_error_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.hlines(y=[results["beta_error_init"]], xmin=[0], xmax=[ITER_NUMS[-1]], colors=["r"])
    plt.xlabel("Iteration Number")
    plt.ylabel("Error in Beta Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"beta.svg"))
    plt.close()

    plt.violinplot([results[f"total_detects_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
    plt.xlabel("Iteration Number")
    plt.ylabel("Total Positive Detections")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"total_detects.svg"))
    plt.close()

    plt.violinplot([results[f"prop_detects_{i}"] for i in ITER_NUMS],
                   ITER_NUMS,
                   showmeans=False,
                   showmedians=True)
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

    # compute flux contour lines
    areax, velocityx = np.meshgrid(np.linspace(0.05, 1.0, 100), np.linspace(0.05, 1.5, 100))
    q0 = (areax / np.pi * velocityx)
    m0 = (q0 * velocityx)
    f0 = 9.81 * 10**(-4) * (300 - pacific_sp_T(0)) * q0
    combos = []
    for i in range(len(EXPS_TO_PROCESS)):
        axp = [results[f"area_mle_{num}"][i] for num in ITER_NUMS]
        vxp = [results[f"velocity_mle_{num}"][i] for num in ITER_NUMS]
        combos.append((([area_mle[0]] + axp),
                       ([velocity_mle[0]] + vxp)))
    fig, ax = plt.subplots(1, 3, sharex=True, sharey=True)
    qs = ax[0].contour(areax, velocityx, q0, 10)
    # qst = ax[0].contour(areax, velocityx, q0, levels=[env.q0])
    ax[0].clabel(qs, inline=1, fontsize=10)
    # ax[1].clabel(qst, inline=1, fontsize=10)
    for pair in combos:
        ax[0].plot(pair[0], pair[1])
    ax[0].scatter(area_mle[0], velocity_mle[0], color="red")
    ax[0].scatter(env.a0, env.v0, color="green")
    ax[0].set_xlabel("Area")
    ax[0].set_ylabel("Velocity")
    ax[0].set_title("Heat Flux")

    ms = ax[1].contour(areax, velocityx, m0, 10)
    # mst = ax[1].contour(areax, velocityx, m0, levels=[env.m0])
    ax[1].clabel(ms, inline=1, fontsize=10)
    # ax[1].clabel(mst, inline=1, fontsize=10)
    for pair in combos:
        ax[1].plot(pair[0], pair[1])
    ax[1].scatter(area_mle[0], velocity_mle[0], color="red")
    ax[1].scatter(env.a0, env.v0, color="green")
    ax[1].set_xlabel("Area")
    ax[1].set_title("Momentum Flux")

    bs = ax[2].contour(areax, velocityx, f0, 10)
    # bst = ax[2].contour(areax, velocityx, f0, levels=[env.f0])
    ax[2].clabel(bs, inline=1, fontsize=10)
    # ax[2].clabel(bst, inline=1, fontsize=10)
    for pair in combos:
        ax[2].plot(pair[0], pair[1])
    ax[2].scatter(area_mle[0], velocity_mle[0], color="red")
    ax[2].scatter(env.a0, env.v0, color="green")
    ax[2].set_xlabel("Area")
    ax[2].set_title("Buoyancy Flux")
    plt.show()
    fig.savefig(os.path.join(SAVE_DIRECTORY, f"flux.svg"))
    plt.close()

    # Save the results
    filepath = os.path.join(SAVE_DIRECTORY, f"results.json")
    j_fp = open(filepath, 'w')
    json.dump(results, j_fp)
    j_fp.close()
