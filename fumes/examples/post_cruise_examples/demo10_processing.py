"""Iterates through paper results and computes metrics"""

import os
import json
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

from fumes.environment.mtt import CrossflowMTT
from fumes.environment.extent import Extent
from fumes.model.parameter import ParameterKDE
from fumes.model.mtt import Crossflow
from fumes.environment.current import CurrHead, CurrMag
from fumes.environment.profile import Profile

from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, curfunc, headfunc
from fumes.utils.save_mission import load_experiment_json


def set_axis_style(ax, labels):
    ax.get_xaxis().set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    ax.set_xlabel('Experiment')


def make_violinplot(numrows, data, labels, ylabel, savename, with_hline=None):
    fig, ax = plt.subplots(numrows, 1, sharey=True)
    for i in range(numrows):
        ax[i].violinplot(data[i],
                         showmeans=False,
                         showmedians=True)
        set_axis_style(ax[i], labels[i])
        ax[i].set_ylabel(ylabel)
        if with_hline is not None:
            ax[i].hlines([with_hline], [0], [len(labels[i]) + 1], color="red")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"{savename}.svg"))
    plt.close()


SAVE_DIRECTORY = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/paperplots_150120100")
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
EXP_PREFIX = ["cloud_map100_iterativeplans_seed",
              "cloud_map120_iterativeplans_seed", "cloud_map150_iterativeplans_seed"]
EXPS_TO_PROCESS = [["227", "244", "260", "541", "648", "670", "809", "986", "658", "572"],  # 100
                   ["37", "134", "378", "447", "693", "780", "884", "938", "950", "32"],  # 120
                   ["45", "52", "279", "336", "382", "450", "575", "610", "699", "702"]]  # 150
LABELS = ["100m", "120m", "150m"]
ITER_NUMS = [0, 1, 2]

# set number of samples for prediction generation
NUM_FORECAST_SAMPS = 10
RANDOM_SEED = 219
np.random.seed(RANDOM_SEED)

# set the z-axis for computing error
QUERYZ = [100., 120., 150.]

# set the horizontal resolution for computing error
EXTENT = Extent(xrange=(-100., 500.),
                xres=10,
                yrange=(-100., 500.),
                yres=10,
                zrange=(0, 200),
                zres=2,
                global_origin=(0., 0., 0.))

# initial inference targets
s = np.linspace(0, 500, 100)  # distance to integrate over
z = np.linspace(0, 200, 100)  # height to integrate over
tprof = Profile(z, pacific_sp_T(z))  # function that describes background temp
sprof = Profile(z, pacific_sp_S(z))  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T

v0_inf = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(
    np.random.uniform(0.05, 1.5, 5000)[:, np.newaxis])
v0_prop = sp.stats.norm(loc=0, scale=0.1)
v0_param = ParameterKDE(v0_inf, v0_prop, limits=(0.01, 1.55))
v0_baseline_samps = v0_param.sample(NUM_FORECAST_SAMPS)

a0_inf = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(
    np.random.uniform(0.05, 0.95, 5000)[:, np.newaxis])
a0_prop = sp.stats.norm(loc=0, scale=0.1)
a0_param = ParameterKDE(a0_inf, a0_prop, limits=(0.01, 1.0))
a0_baseline_samps = a0_param.sample(NUM_FORECAST_SAMPS)

alph_inf = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(
    np.random.uniform(0.11, 0.21, 5000)[:, np.newaxis])
alph_prop = sp.stats.norm(loc=0, scale=0.05)
alph_param = ParameterKDE(alph_inf, alph_prop, limits=(0.1, 0.22))
alph_baseline_samps = alph_param.sample(NUM_FORECAST_SAMPS)

bet_inf = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(
    np.random.uniform(0.11, 0.21, 5000)[:, np.newaxis])
bet_prop = sp.stats.norm(loc=0, scale=0.05)
bet_param = ParameterKDE(bet_inf, bet_prop, limits=(0.1, 0.22))
bet_baseline_samps = bet_param.sample(NUM_FORECAST_SAMPS)

if __name__ == "__main__":
    results = {"exp_prefix": EXP_PREFIX,
               "exp_seeds": EXPS_TO_PROCESS,
               "exp_labels": LABELS,
               "iters": ITER_NUMS}

    # get the environment (since the same, just need to do this once)
    exemplar_params = load_experiment_json(EXP_PREFIX[0] + EXPS_TO_PROCESS[0][0], ITER_NUMS[0])
    times = np.asarray(exemplar_params["simulation_params"]["times"])
    snap_times = np.unique(np.round(times / 3600.))
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

    # set the baseline
    # set initialized environments
    results["base_iou"] = []
    results["base_ioa"] = []
    results["base_rmse"] = []
    results["base_ce"] = []
    results["base_area"] = a0_baseline_samps.flatten().tolist()
    results["base_velocity"] = v0_baseline_samps.flatten().tolist()
    results["base_alpha"] = alph_baseline_samps.flatten().tolist()
    results["base_beta"] = bet_baseline_samps.flatten().tolist()

    for v, a, alph, bet in zip(v0_baseline_samps, a0_baseline_samps, alph_baseline_samps, bet_baseline_samps):
        mod = CrossflowMTT(extent=EXTENT,
                           plume_loc=(0, 0, 0),
                           s=s,
                           tprof=pacific_sp_T,
                           sprof=pacific_sp_S,
                           rhoprof=eos_rho,
                           headfunc=headfunc,
                           curfunc=curfunc,
                           density=eos_rho(300, 34.608),
                           temp=300,
                           salt=34.608,
                           lam=1,
                           vex=v,
                           area=a,
                           entrainment=(alph, bet)
                           )
        ious = []
        ioas = []
        ces = []
        rmses = []
        for t in snap_times:
            inner_env_obs = []
            inner_mod_obs = []
            env_samps = env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
            for samp in env_samps:
                inner_env_obs.append(samp.flatten() > 1e-5)
            mod_samps = mod.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
            for samp in mod_samps:
                inner_mod_obs.append(samp.flatten() > 1e-5)
            obs_in_both = (np.asarray(inner_env_obs).flatten().astype(float) +
                           np.asarray(inner_mod_obs).flatten().astype(float))
            intersection = len(obs_in_both[obs_in_both >= 2])
            iarea = np.sum(inner_env_obs)
            union = len(obs_in_both[obs_in_both >= 1])
            ious.append(float(intersection / union))
            ioas.append(float(intersection / iarea))
            rmses.append(np.sqrt(np.mean((np.asarray(inner_env_obs).flatten().astype(
                float) - np.asarray(inner_mod_obs).flatten().astype(float))**2)))
            ces.append(np.sum(np.fabs(np.asarray(inner_env_obs).flatten().astype(
                float) - np.asarray(inner_mod_obs).flatten().astype(float))))
        iou = np.mean(ious)
        ioa = np.mean(ioas)
        rmse = np.mean(rmses)
        ce = np.mean(ces)
        results["base_rmse"].append(rmse)
        results["base_ce"].append(ce)
        results["base_iou"].append(iou)
        results["base_ioa"].append(ioa)
    print("Initialized model...")
    print("RMSE: ", np.mean(results["base_rmse"]))
    print("Classificaton Error: ", np.mean(results["base_ce"]))
    print("IOU: ", np.mean(results["base_iou"]))

    # compute error in the params of interest
    results["base_velocity_error"] = (env.v0 - np.mean(v0_baseline_samps)) / env.v0
    results["base_area_error"] = (env.a0 - np.mean(a0_baseline_samps)) / env.a0
    results["base_alpha_error"] = (
        env.entrainment[0] - np.mean(alph_baseline_samps)) / env.entrainment[0]
    results["base_beta_error"] = (
        env.entrainment[1] - np.mean(bet_baseline_samps)) / env.entrainment[1]
    print("Area Error: ", results[f"base_area_error"])
    print("Velocity Error: ", results[f"base_velocity_error"])
    print("Alpha Error: ", results[f"base_alpha_error"])
    print("Beta Error: ", results[f"base_beta_error"])
    print("---")

    for iter in ITER_NUMS:
        for pref, exps, lab in zip(EXP_PREFIX, EXPS_TO_PROCESS, LABELS):
            # initialize storage
            results[f"rmse_{iter}_{lab}"] = []
            results[f"class_error_{iter}_{lab}"] = []
            results[f"iou_{iter}_{lab}"] = []
            results[f"ioa_{iter}_{lab}"] = []
            results[f"area_{iter}_{lab}"] = []
            results[f"area_error_{iter}_{lab}"] = []
            results[f"velocity_{iter}_{lab}"] = []
            results[f"velocity_error_{iter}_{lab}"] = []
            results[f"alpha_{iter}_{lab}"] = []
            results[f"alpha_error_{iter}_{lab}"] = []
            results[f"beta_{iter}_{lab}"] = []
            results[f"beta_error_{iter}_{lab}"] = []
            results[f"total_detects_{iter}_{lab}"] = []
            results[f"prop_detects_{iter}_{lab}"] = []
            results[f"spatial_{iter}_{lab}"] = []
            results[f"temporal_{iter}_{lab}"] = []

            for exp in exps:
                print(lab, exp, iter)
                exp_json = load_experiment_json(pref + exp, iter)

                # get the model distributions
                mod_params = exp_json["model_params"]
                mod = CrossflowMTT(extent=EXTENT,
                                   plume_loc=(0, 0, 0),
                                   s=s,
                                   tprof=pacific_sp_T,
                                   sprof=pacific_sp_S,
                                   rhoprof=eos_rho,
                                   headfunc=headfunc,
                                   curfunc=curfunc,
                                   density=eos_rho(300, 34.608),
                                   temp=300,
                                   salt=34.608,
                                   lam=1,
                                   vex=mod_params["model_learned_params"]["velocity_mle"],
                                   area=mod_params["model_learned_params"]["area_mle"],
                                   entrainment=(mod_params["model_learned_params"]["entrainment_alpha_mle"],
                                                mod_params["model_learned_params"]["entrainment_beta_mle"]),
                                   )

                ious = []
                ioas = []
                ces = []
                rmses = []
                for t in snap_times:
                    inner_env_obs = []
                    inner_mod_obs = []
                    env_samps = env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
                    for samp in env_samps:
                        inner_env_obs.append(samp.flatten() > 1e-5)
                    mod_samps = mod.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
                    for samp in mod_samps:
                        inner_mod_obs.append(samp.flatten() > 1e-5)
                    obs_in_both = (np.asarray(inner_env_obs).flatten().astype(float) +
                                   np.asarray(inner_mod_obs).flatten().astype(float))
                    intersection = len(obs_in_both[obs_in_both >= 2])
                    iarea = np.sum(inner_env_obs)
                    union = len(obs_in_both[obs_in_both >= 1])
                    ious.append(float(intersection / union))
                    ioas.append(float(intersection / iarea))
                    rmses.append(np.sqrt(np.mean((np.asarray(inner_env_obs).flatten().astype(
                        float) - np.asarray(inner_mod_obs).flatten().astype(float))**2)))
                    ces.append(np.sum(np.fabs(np.asarray(inner_env_obs).flatten().astype(
                        float) - np.asarray(inner_mod_obs).flatten().astype(float))))
                iou = np.mean(ious)
                rmse = np.mean(rmses)
                ce = np.mean(ces)
                ioa = np.mean(ioas)
                results[f"rmse_{iter}_{lab}"].append(rmse)
                results[f"class_error_{iter}_{lab}"].append(ce)
                results[f"iou_{iter}_{lab}"].append(iou)
                results[f"ioa_{iter}_{lab}"].append(ioa)

                results[f"area_{iter}_{lab}"].append(mod.a0)
                results[f"velocity_{iter}_{lab}"].append(mod.v0)
                results[f"alpha_{iter}_{lab}"].append(mod.entrainment[0])
                results[f"beta_{iter}_{lab}"].append(mod.entrainment[1])

                results[f"area_error_{iter}_{lab}"].append((env.a0 - mod.a0) / env.a0)
                results[f"velocity_error_{iter}_{lab}"].append((env.v0 - mod.v0) / env.v0)
                results[f"alpha_error_{iter}_{lab}"].append(
                    (env.entrainment[0] - mod.entrainment[0]) / env.entrainment[0])
                results[f"beta_error_{iter}_{lab}"].append(
                    (env.entrainment[1] - mod.entrainment[1]) / env.entrainment[1])

                total_detects = exp_json["experiment_params"]["total_in_plume_samples"]
                prop_detects = exp_json["experiment_params"]["portion_in_plume_samples"]
                results[f"total_detects_{iter}_{lab}"].append(total_detects)
                results[f"prop_detects_{iter}_{lab}"].append(prop_detects)

                # get simulated observations
                sim_obs = np.asarray(exp_json["simulation_params"]["obs"])
                sim_coords = exp_json["simulation_params"]["coords"]

                # grab spatial extent (furthest detecton over furthest distance)
                distances = np.asarray([np.sqrt((coord[0])**2 + (coord[1])**2)
                                       for coord in sim_coords])
                max_dist = np.nanmax(distances)
                max_detect_dist = np.nanmax(distances[sim_obs > 1e-5])
                spatial_extent = max_detect_dist / max_dist
                results[f"spatial_{iter}_{lab}"].append(spatial_extent)

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
                results[f"temporal_{iter}_{lab}"].append(temporal_util)

    # plot "pretty" composite graphs of metrics
    # model performance plots
    rmse_violin = [[results["base_rmse"]] for i in LABELS]
    ce_violin = [[results["base_ce"]] for i in LABELS]
    iou_violin = [[results["base_iou"]] for i in LABELS]
    ioa_violin = [[results["base_ioa"]] for i in LABELS]
    area_violin = [[results["base_area"]] for i in LABELS]
    velocity_violin = [[results["base_velocity"]] for i in LABELS]
    alpha_violin = [[results["base_alpha"]] for i in LABELS]
    beta_violin = [[results["base_beta"]] for i in LABELS]
    violin_labels = [["Baseline"] for i in LABELS]
    for j, l in enumerate(LABELS):
        for i in ITER_NUMS[0:-1]:
            rmse_violin[j].append(results[f"rmse_{i}_{l}"])
            ce_violin[j].append(results[f"class_error_{i}_{l}"])
            iou_violin[j].append(results[f"iou_{i}_{l}"])
            ioa_violin[j].append(results[f"ioa_{i}_{l}"])
            area_violin[j].append(results[f"area_{i}_{l}"])
            velocity_violin[j].append(results[f"velocity_{i}_{l}"])
            alpha_violin[j].append(results[f"alpha_{i}_{l}"])
            beta_violin[j].append(results[f"beta_{i}_{l}"])
            violin_labels[j].append(f"{l} {i}")
    
    make_violinplot(len(EXPS_TO_PROCESS), rmse_violin, violin_labels, "RMSE", "rmse")
    make_violinplot(len(EXPS_TO_PROCESS), ce_violin, violin_labels, "Class Error", "ce")
    make_violinplot(len(EXPS_TO_PROCESS), iou_violin, violin_labels, "IoU", "iou")
    make_violinplot(len(EXPS_TO_PROCESS), ioa_violin, violin_labels, "IoA", "ioa")    
    make_violinplot(len(EXPS_TO_PROCESS), area_violin,
                    violin_labels, "area", "area", with_hline=env.a0)
    make_violinplot(len(EXPS_TO_PROCESS), velocity_violin, violin_labels,
                    "velocity", "velocity", with_hline=env.v0)
    make_violinplot(len(EXPS_TO_PROCESS), alpha_violin, violin_labels,
                    "alpha", "alpha", with_hline=env.entrainment[0])
    make_violinplot(len(EXPS_TO_PROCESS), beta_violin, violin_labels,
                    "beta", "beta", with_hline=env.entrainment[1])
    
    # phortex performance plots
    total_detects = [[] for i in ITER_NUMS]
    prop_detects = [[] for i in ITER_NUMS]
    spat_util = [[] for i in ITER_NUMS]
    temp_util = [[] for i in ITER_NUMS]
    violin_labels = [[] for i in ITER_NUMS]
    for j, l in enumerate(LABELS):
        for i in ITER_NUMS:
            total_detects[j].append(results[f"total_detects_{i}_{l}"])
            prop_detects[j].append(results[f"prop_detects_{i}_{l}"])
            spat_util[j].append(results[f"spatial_{i}_{l}"])
            temp_util[j].append(results[f"temporal_{i}_{l}"])
            violin_labels[j].append(f"{l} {i}")

    make_violinplot(len(EXPS_TO_PROCESS), total_detects,
                    violin_labels, "total_detects", "total_detects")
    make_violinplot(len(EXPS_TO_PROCESS), prop_detects,
                    violin_labels, "prop_detects", "prop_detects")
    make_violinplot(len(EXPS_TO_PROCESS), spat_util, violin_labels,
                    "spatial_utility", "spatial_utility")
    make_violinplot(len(EXPS_TO_PROCESS), temp_util, violin_labels,
                    "temporal_utility", "temporal_utility")

    # Save the results
    filepath = os.path.join(SAVE_DIRECTORY, f"results.json")
    j_fp = open(filepath, 'w')
    json.dump(results, j_fp)
    j_fp.close()
