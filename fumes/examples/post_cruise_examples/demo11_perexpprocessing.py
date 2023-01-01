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


def make_violinplot(numrows, data, labels, ylabel, savename):
    fig, ax = plt.subplots(numrows, 1, sharey=True)
    for i in range(numrows):
        ax[i].violinplot(data[i],
                         showmeans=False,
                         showmedians=True)
        set_axis_style(ax[i], labels[i])
        ax[i].set_ylabel(ylabel)
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"{savename}.svg"))
    plt.close()


SAVE_DIRECTORY = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/summary_expchains_120")
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
# EXP_PREFIX = "cloud_naivephumes_iterativeplans_seed"  # 120m
# EXP_PREFIX = "cloud_phortexphumes_iterativeplans_seed"
EXP_PREFIX = ["cloud_phortexphumes_iterativeplans_seed", "cloud_naivephumes_iterativeplans_seed"]
EXPS_TO_PROCESS = [["20", "62", "125", "291", "139", "495", "592", "617", "301", "196"],  # phortex120
                   ["37", "154", "225", "289", "479", "639", "685", "707", "709", "958"]]  # naive120
# NAIVE_TO_PROCESS = ["553", "51", "124", "703", "214", "539", "369", "195"]  # naive100
LABELS = ["PHORTEX", "NAIVE"]
ITER_NUMS = [0]

# set number of samples for prediction generation
NUM_FORECAST_SAMPS = 100

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
        all_env_obs = []
        all_mod_obs = []
        for t in snap_times:
            env_samps = env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
            for samp in env_samps:
                all_env_obs.append(samp.flatten() > 1e-5)
            mod_samps = mod.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
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
        results["base_rmse"].append(rmse)
        results["base_ce"].append(class_error)
        results["base_iou"].append(iou)
    print("Initialized model...")
    print("RMSE: ", np.mean(results["base_rmse"]))
    print("Classificaton Error: ", np.mean(results["base_ce"]))
    print("IOU: ", np.mean(results["base_iou"]))
    print("----")

    for iter in ITER_NUMS:
        for pref, exps, lab in zip(EXP_PREFIX, EXPS_TO_PROCESS, LABELS):
            for exp in exps:
                # initialize storage
                results[f"rmse_{iter}_{lab}{exp}"] = []
                results[f"class_error_{iter}_{lab}{exp}"] = []
                results[f"iou_{iter}_{lab}{exp}"] = []
                results[f"heat_flux_error_{iter}_{lab}{exp}"] = []
                results[f"momentum_flux_error_{iter}_{lab}{exp}"] = []
                results[f"buoyancy_flux_error_{iter}_{lab}{exp}"] = []
                results[f"area_{iter}_{lab}{exp}"] = []
                results[f"velocity_{iter}_{lab}{exp}"] = []
                results[f"alpha_{iter}_{lab}{exp}"] = []
                results[f"beta_{iter}_{lab}{exp}"] = []

                print(lab, exp, iter)
                exp_json = load_experiment_json(pref + exp, iter)

                # get the model distributions
                mod_params = exp_json["model_params"]
                chain = np.asarray(mod_params["model_update_procedure"]["chain_samples"])
                kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(chain[:][150:])
                samples = kde.sample(NUM_FORECAST_SAMPS)
                results[f"area_{iter}_{lab}{exp}"] = samples[:, 3].flatten().tolist()
                results[f"velocity_{iter}_{lab}{exp}"] = samples[:, 2].flatten().tolist()
                results[f"alpha_{iter}_{lab}{exp}"] = samples[:, 0].flatten().tolist()
                results[f"beta_{iter}_{lab}{exp}"] = samples[:, 1].flatten().tolist()

                for samp in samples:
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
                                       vex=samp[2],
                                       area=samp[3],
                                       entrainment=(samp[0], samp[1])
                                       )
                    all_mod_obs = []
                    for t in snap_times:
                        mod_samps = mod.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
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
                    results[f"rmse_{iter}_{lab}{exp}"].append(rmse)
                    results[f"class_error_{iter}_{lab}{exp}"].append(class_error)
                    results[f"iou_{iter}_{lab}{exp}"].append(iou)
                print("RMSE: ", np.mean(results[f"rmse_{iter}_{lab}{exp}"]))
                print("Classificaton Error: ", np.mean(results[f"class_error_{iter}_{lab}{exp}"]))
                print("IOU: ", np.mean(results[f"iou_{iter}_{lab}{exp}"]))
                print("-----")

    # plot "pretty" composite graphs of metrics
    rmse_violin = [[results["base_rmse"]], [results["base_rmse"]]]
    class_error_violin = [[results["base_ce"]], [results["base_ce"]]]
    iou_violin = [[results["base_iou"]], [results["base_iou"]]]
    area_violin = [[results["base_area"]], [results["base_area"]]]
    velocity_violin = [[results["base_velocity"]], [results["base_velocity"]]]
    alpha_violin = [[results["base_alpha"]], [results["base_alpha"]]]
    beta_violin = [[results["base_beta"]], [results["base_beta"]]]
    violin_labels = [["Baseline"], ["Baseline"]]
    for i, (exps, lab) in enumerate(zip(EXPS_TO_PROCESS, LABELS)):
        for exp in exps:
            rmse_violin[i].append(results[f"rmse_0_{lab}{exp}"])
            class_error_violin[i].append(results[f"class_error_0_{lab}{exp}"])
            iou_violin[i].append(results[f"iou_0_{lab}{exp}"])
            area_violin[i].append(results[f"area_0_{lab}{exp}"])
            velocity_violin[i].append(results[f"velocity_0_{lab}{exp}"])
            alpha_violin[i].append(results[f"alpha_0_{lab}{exp}"])
            beta_violin[i].append(results[f"beta_0_{lab}{exp}"])
            violin_labels[i].append(f"{lab} {exp}")

    make_violinplot(len(EXPS_TO_PROCESS), rmse_violin, violin_labels, "RMSE", "rmse")
    make_violinplot(len(EXPS_TO_PROCESS), iou_violin, violin_labels, "IoU", "iou")
    make_violinplot(len(EXPS_TO_PROCESS), class_error_violin,
                    violin_labels, "Class Error", "class_error")
    make_violinplot(len(EXPS_TO_PROCESS), area_violin, violin_labels, "area", "area")
    make_violinplot(len(EXPS_TO_PROCESS), velocity_violin, violin_labels, "velocity", "velocity")
    make_violinplot(len(EXPS_TO_PROCESS), alpha_violin, violin_labels, "alpha", "alpha")
    make_violinplot(len(EXPS_TO_PROCESS), beta_violin, violin_labels, "beta", "beta")

    # Save the results
    filepath = os.path.join(SAVE_DIRECTORY, f"results.json")
    j_fp = open(filepath, 'w')
    json.dump(results, j_fp)
    j_fp.close()
