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

SAVE_DIRECTORY = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/summary_chains_120100")
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
# EXP_PREFIX = "cloud_naivephumes_iterativeplans_seed"  # 120m
# EXP_PREFIX = "cloud_phortexphumes_iterativeplans_seed"
EXP_PREFIX = "cloud_"
PHORTEX_PREFIX = "naivephumes_iterativeplans_seed"
NAIVE_PREFIX = "naivephumes_iterativeplans_seed"
PHORTEX_TO_PROCESS = ["20", "62", "125", "291", "139", "495", "592", "617", "301", "196"] #phortex120
PHORTEX_TO_PROCESS = ["553", "51", "124", "703", "214", "539", "369", "195"]  # naive100
NAIVE_TO_PROCESS = ["37", "154", "225", "289", "479", "639", "685", "707", "709", "958"]  # naive120
LABELS = ["NAIVE100", "NAIVE120"]
ITER_NUMS = [0]

# set number of samples for prediction generation
NUM_FORECAST_SAMPS = 5

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

a0_inf = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(
    np.random.uniform(0.05, 0.95, 5000)[:, np.newaxis])
a0_prop = sp.stats.norm(loc=0, scale=0.1)
a0_param = ParameterKDE(a0_inf, a0_prop, limits=(0.01, 1.0))

alph_inf = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(
    np.random.uniform(0.11, 0.21, 5000)[:, np.newaxis])
alph_prop = sp.stats.norm(loc=0, scale=0.05)
alph_param = ParameterKDE(alph_inf, alph_prop, limits=(0.1, 0.22))

bet_inf = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(
    np.random.uniform(0.11, 0.21, 5000)[:, np.newaxis])
bet_prop = sp.stats.norm(loc=0, scale=0.05)
bet_param = ParameterKDE(bet_inf, bet_prop, limits=(0.1, 0.22))

duration = 3600 * 12
training_t = np.linspace(0, duration + 1, 100)
curmag = CurrMag(training_t / 3600. % 24., curfunc(None, training_t) + np.random.normal(0, 0.01, training_t.shape),
                 training_iter=500, learning_rate=0.5)
curhead = CurrHead(training_t / 3600. % 24., headfunc(training_t) * 180. / np.pi + np.random.normal(0, 0.01, training_t.shape),
                   training_iter=500, learning_rate=0.5)

# set initialized model
BASELINE = Crossflow(plume_loc=(0, 0, 0),
                     extent=EXTENT,
                     s=s,
                     tprof=tprof,
                     sprof=sprof,
                     rhoprof=rhoprof,
                     vex=v0_param,
                     area=a0_param,
                     density=eos_rho(300, 34.608),
                     salt=34.608,
                     temp=300,
                     curfunc=curmag,
                     headfunc=curhead,
                     E=(alph_param, bet_param))
area_map = [BASELINE.odesys.a0]
velocity_map = [BASELINE.odesys.v0]
alpha_map = [BASELINE.odesys.entrainment[0]]
beta_map = [BASELINE.odesys.entrainment[1]]

if __name__ == "__main__":
    results = {"phortex_exp_prefix": EXP_PREFIX + PHORTEX_PREFIX,
               "naive_exp_prefix": EXP_PREFIX + NAIVE_PREFIX,
               "phortex_exp_seeds": PHORTEX_TO_PROCESS,
               "naive_exp_seeds": NAIVE_TO_PROCESS,
               "iters": ITER_NUMS}

    for iter in ITER_NUMS:
        for lab in LABELS:
            # initialize storage
            results[f"rmse_{iter}_{lab}"] = []
            results[f"class_error_{iter}_{lab}"] = []
            results[f"iou_{iter}_{lab}"] = []
            results[f"total_detects_{iter}_{lab}"] = []
            results[f"prop_detects_{iter}_{lab}"] = []
            results[f"heat_flux_error_{iter}_{lab}"] = []
            results[f"momentum_flux_error_{iter}_{lab}"] = []
            results[f"buoyancy_flux_error_{iter}_{lab}"] = []
            results[f"area_map_{iter}_{lab}"] = []
            results[f"area_samples_{iter}_{lab}"] = []
            results[f"area_error_{iter}_{lab}"] = []
            results[f"velocity_map_{iter}_{lab}"] = []
            results[f"velocity_samples_{iter}_{lab}"] = []
            results[f"velocity_error_{iter}_{lab}"] = []
            results[f"alpha_map_{iter}_{lab}"] = []
            results[f"alpha_error_{iter}_{lab}"] = []
            results[f"beta_map_{iter}_{lab}"] = []
            results[f"beta_error_{iter}_{lab}"] = []

    # get the environment (since the same, just need to do this once)
    exemplar_params = load_experiment_json(
        EXP_PREFIX + PHORTEX_PREFIX + PHORTEX_TO_PROCESS[0], ITER_NUMS[0])
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

    #######
    # compute baseline rmse, class_error
    #######
    times = np.asarray(exemplar_params["simulation_params"]["times"])
    snap_times = np.unique(np.round(times / 3600.))
    print("Computing...")
    all_env_obs = []
    all_mod_obs = []
    for t in snap_times:
        env_samps = env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
        for samp in env_samps:
            all_env_obs.append(samp.flatten() > 1e-5)
        mod_samps = BASELINE.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
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

    ######
    # compute the flux error
    ######
    init_heatflux = (env.q0 - BASELINE.odesys.q0) / env.q0
    init_momflux = (env.m0 - BASELINE.odesys.m0) / env.m0
    init_buoyflux = (env.f0 - BASELINE.odesys.f0) / env.f0
    results["heat_flux_error_init"] = (env.q0 - BASELINE.odesys.q0) / env.q0
    results["momentum_flux_error_init"] = (env.m0 - BASELINE.odesys.m0) / env.m0
    results["buoyancy_flux_error_init"] = (env.f0 - BASELINE.odesys.f0) / env.f0
    print("Heat flux error ", results["heat_flux_error_init"])
    print("Momentum flux error ", results["momentum_flux_error_init"])
    print("Buoyancy flux error ", results["buoyancy_flux_error_init"])

    # compute error in the params of interest
    results["velocity_error_init"] = (env.v0 - BASELINE.odesys.v0) / env.v0
    results["area_error_init"] = (env.a0 - BASELINE.odesys.a0) / env.a0
    results["alpha_error_init"] = (
        env.entrainment[0] - BASELINE.odesys.entrainment[0]) / env.entrainment[0]
    results["beta_error_init"] = (
        env.entrainment[1] - BASELINE.odesys.entrainment[1]) / env.entrainment[1]
    print("Area Error: ", results[f"area_error_init"])
    print("Velocity Error: ", results[f"velocity_error_init"])
    print("Alpha Error: ", results[f"alpha_error_init"])
    print("Beta Error: ", results[f"beta_error_init"])
    print("---")

    # set up chain plots
    chain_area = []
    chain_velocity = []
    chain_alpha = []
    chain_beta = []

    # Read in each simulation json
    for exp in NAIVE_TO_PROCESS:
        lab = LABELS[1]
        for iter in ITER_NUMS:
            print(lab, exp, iter)
            exp_json = load_experiment_json(EXP_PREFIX + NAIVE_PREFIX + exp, iter)
            times = np.asarray(exp_json["simulation_params"]["times"])
            snap_times = np.unique(np.round(times / 3600.))

            # get the trained model
            mod_params = exp_json["model_params"]
            chain = np.asarray(mod_params["model_update_procedure"]["chain_samples"])
            chain_area.append(chain[:, 3][150:])
            chain_velocity.append(chain[:, 2][150:])
            chain_alpha.append(chain[:, 0][150:])
            chain_beta.append(chain[:, 1][150:])

            kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(chain[:][150:])
            height = np.exp(kde.score_samples(chain[:][150:]))
            maps = chain[:][150:][np.argmax(height)]

            # MAP estimate from params
            mod_env = CrossflowMTT(extent=EXTENT,
                                   plume_loc=mod_params["model_fixed_params"]["plume_loc"],
                                   s=mod_params["model_fixed_params"]["s"],
                                   curfunc=curfunc,
                                   headfunc=headfunc,
                                   tprof=pacific_sp_T,
                                   sprof=pacific_sp_S,
                                   rhoprof=eos_rho,
                                   vex=maps[2],
                                   area=maps[3],
                                   density=mod_params["model_fixed_params"]["density"],
                                   salt=mod_params["model_fixed_params"]["salt"],
                                   temp=mod_params["model_fixed_params"]["temp"],
                                   lam=env_params["lam"],
                                   entrainment=(maps[0], maps[1]))
            results[f"area_map_{iter}_{lab}"].append(mod_env.a0)
            results[f"velocity_map_{iter}_{lab}"].append(mod_env.v0)

            # compute the RMSE between each simulated hour snapshot and the true underlying environment
            # for a random sample of points
            all_env_obs = []
            all_mod_obs = []
            for t in snap_times:
                env_samps = env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
                for samp in env_samps:
                    all_env_obs.append(samp.flatten() > 1e-5)
                mod_samps = mod_env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
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
            results[f"rmse_{iter}_{lab}"].append(rmse)
            results[f"class_error_{iter}_{lab}"].append(class_error)
            results[f"iou_{iter}_{lab}"].append(iou)
            print(f"RMSE: {rmse}")
            print(f"Classificaton Error: {class_error}")
            print(f"IOU: {iou}")

            # get science error
            results[f"heat_flux_error_{iter}_{lab}"] = (env.q0 - mod_env.q0) / env.q0
            results[f"momentum_flux_error_{iter}_{lab}"] = (env.m0 - mod_env.m0) / env.m0
            results[f"buoyancy_flux_error_{iter}_{lab}"] = (env.f0 - mod_env.f0) / env.f0
            print("Heat flux error ", results[f"heat_flux_error_{iter}_{lab}"])
            print("Momentum flux error ", results[f"momentum_flux_error_{iter}_{lab}"])
            print("Buoyancy flux error ", results[f"buoyancy_flux_error_{iter}_{lab}"])

            # get param error
            results[f"velocity_error_{iter}_{lab}"].append((env.v0 - mod_env.v0) / env.v0)
            results[f"area_error_{iter}_{lab}"].append((env.a0 - mod_env.a0) / env.a0)
            results[f"alpha_error_{iter}_{lab}"].append(
                (env.entrainment[0] - mod_env.entrainment[0]) / env.entrainment[0])
            results[f"beta_error_{iter}_{lab}"].append(
                (env.entrainment[1] - mod_env.entrainment[1]) / env.entrainment[1])
            print("Area Error: ", results[f"area_error_{iter}_{lab}"][-1])
            print("Velocity Error: ", results[f"velocity_error_{iter}_{lab}"][-1])
            print("Alpha Error: ", results[f"alpha_error_{iter}_{lab}"][-1])
            print("Beta Error: ", results[f"beta_error_{iter}_{lab}"][-1])

            # grab total samples and proportion samples for each simulation
            total_detects = exp_json["experiment_params"]["total_in_plume_samples"]
            prop_detects = exp_json["experiment_params"]["portion_in_plume_samples"]
            results[f"total_detects_{iter}_{lab}"].append(total_detects)
            results[f"prop_detects_{iter}_{lab}"].append(prop_detects)
            print(f"Total Detections: {total_detects}")
            print(f"Total Prop Detections: {prop_detects}")

            # get simulated observations
            sim_obs = np.asarray(exp_json["simulation_params"]["obs"])
            sim_coords = exp_json["simulation_params"]["coords"]
            print("----")

    # Read in each simulation json
    for exp in PHORTEX_TO_PROCESS:
        lab = LABELS[0]
        for iter in ITER_NUMS:
            print(lab, exp, iter)
            exp_json = load_experiment_json(EXP_PREFIX + PHORTEX_PREFIX + exp, iter)
            times = np.asarray(exp_json["simulation_params"]["times"])
            snap_times = np.unique(np.round(times / 3600.))

            # get the trained model
            mod_params = exp_json["model_params"]
            chain = np.asarray(mod_params["model_update_procedure"]["chain_samples"])
            chain_area.append(chain[:, 3][150:])
            chain_velocity.append(chain[:, 2][150:])
            chain_alpha.append(chain[:, 0][150:])
            chain_beta.append(chain[:, 1][150:])

            kde = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(chain[:][150:])
            height = np.exp(kde.score_samples(chain[:][150:]))
            maps = chain[:][150:][np.argmax(height)]

            # MAP estimate from params
            mod_env = CrossflowMTT(extent=EXTENT,
                                   plume_loc=mod_params["model_fixed_params"]["plume_loc"],
                                   s=mod_params["model_fixed_params"]["s"],
                                   curfunc=curfunc,
                                   headfunc=headfunc,
                                   tprof=pacific_sp_T,
                                   sprof=pacific_sp_S,
                                   rhoprof=eos_rho,
                                   vex=maps[2],
                                   area=maps[3],
                                   density=mod_params["model_fixed_params"]["density"],
                                   salt=mod_params["model_fixed_params"]["salt"],
                                   temp=mod_params["model_fixed_params"]["temp"],
                                   lam=env_params["lam"],
                                   entrainment=(maps[0], maps[1]))
            results[f"area_map_{iter}_{lab}"].append(mod_env.a0)
            results[f"velocity_map_{iter}_{lab}"].append(mod_env.v0)

            # compute the RMSE between each simulated hour snapshot and the true underlying environment
            # for a random sample of points
            all_env_obs = []
            all_mod_obs = []
            for t in snap_times:
                env_samps = env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
                for samp in env_samps:
                    all_env_obs.append(samp.flatten() > 1e-5)
                mod_samps = mod_env.get_snapshot(t=t * 3600., z=QUERYZ, from_cache=False)
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
            results[f"rmse_{iter}_{lab}"].append(rmse)
            results[f"class_error_{iter}_{lab}"].append(class_error)
            results[f"iou_{iter}_{lab}"].append(iou)
            print(f"RMSE: {rmse}")
            print(f"Classificaton Error: {class_error}")
            print(f"IOU: {iou}")

            # get science error
            results[f"heat_flux_error_{iter}_{lab}"] = (env.q0 - mod_env.q0) / env.q0
            results[f"momentum_flux_error_{iter}_{lab}"] = (env.m0 - mod_env.m0) / env.m0
            results[f"buoyancy_flux_error_{iter}_{lab}"] = (env.f0 - mod_env.f0) / env.f0
            print("Heat flux error ", results[f"heat_flux_error_{iter}_{lab}"])
            print("Momentum flux error ", results[f"momentum_flux_error_{iter}_{lab}"])
            print("Buoyancy flux error ", results[f"buoyancy_flux_error_{iter}_{lab}"])

            # get param error
            results[f"velocity_error_{iter}_{lab}"].append((env.v0 - mod_env.v0) / env.v0)
            results[f"area_error_{iter}_{lab}"].append((env.a0 - mod_env.a0) / env.a0)
            results[f"alpha_error_{iter}_{lab}"].append(
                (env.entrainment[0] - mod_env.entrainment[0]) / env.entrainment[0])
            results[f"beta_error_{iter}_{lab}"].append(
                (env.entrainment[1] - mod_env.entrainment[1]) / env.entrainment[1])
            print("Area Error: ", results[f"area_error_{iter}_{lab}"][-1])
            print("Velocity Error: ", results[f"velocity_error_{iter}_{lab}"][-1])
            print("Alpha Error: ", results[f"alpha_error_{iter}_{lab}"][-1])
            print("Beta Error: ", results[f"beta_error_{iter}_{lab}"][-1])

            # grab total samples and proportion samples for each simulation
            total_detects = exp_json["experiment_params"]["total_in_plume_samples"]
            prop_detects = exp_json["experiment_params"]["portion_in_plume_samples"]
            results[f"total_detects_{iter}_{lab}"].append(total_detects)
            results[f"prop_detects_{iter}_{lab}"].append(prop_detects)
            print(f"Total Detections: {total_detects}")
            print(f"Total Prop Detections: {prop_detects}")

            # get simulated observations
            sim_obs = np.asarray(exp_json["simulation_params"]["obs"])
            sim_coords = exp_json["simulation_params"]["coords"]
            print("----")


    def set_axis_style(ax, labels):
        ax.get_xaxis().set_tick_params(direction='out')
        ax.xaxis.set_ticks_position('bottom')
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_xlim(0.25, len(labels) + 0.75)
        ax.set_xlabel('Trajectory')

    # plot "pretty" composite graphs of metrics
    plt.violinplot([results[f"rmse_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("RMSE")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"rmse.svg"))
    plt.close()

    plt.violinplot([results[f"class_error_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Classification Error")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"class_error.svg"))
    plt.close()

    plt.violinplot([results[f"iou_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Intersection over Union")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"iou.svg"))
    plt.close()

    plt.violinplot([results[f"area_error_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Error in Area Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"area.svg"))
    plt.close()

    plt.violinplot([results[f"velocity_error_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Error in Velocity Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"velocity.svg"))
    plt.close()

    plt.violinplot([results[f"alpha_error_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Error in Alpha Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"alpha.svg"))
    plt.close()

    plt.violinplot([results[f"beta_error_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Error in Beta Estimate")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"beta.svg"))
    plt.close()

    plt.violinplot([results[f"total_detects_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Total Positive Detections")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"total_detects.svg"))
    plt.close()

    plt.violinplot([results[f"prop_detects_0_{lab}"] for lab in LABELS],
                   showmeans=False,
                   showmedians=True)
    set_axis_style(plt.gca(), LABELS)
    plt.ylabel("Proportion Positive Detections")
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"prop_detects.svg"))
    plt.close()

    # compute flux contour lines
    areax, velocityx = np.meshgrid(np.linspace(0.05, 1.0, 100), np.linspace(0.05, 1.5, 100))
    q0 = (areax / np.pi * velocityx)
    m0 = (q0 * velocityx)
    f0 = 9.81 * 10**(-4) * (300 - pacific_sp_T(0)) * q0
    pcombos = []
    ncombos = []
    for i in range(len(PHORTEX_TO_PROCESS)):
        axp = [results[f"area_map_{num}_{LABELS[0]}"][i] for num in ITER_NUMS]
        vxp = [results[f"velocity_map_{num}_{LABELS[0]}"][i] for num in ITER_NUMS]
        pcombos.append((([area_map[0]] + axp),
                       ([velocity_map[0]] + vxp)))
    for i in range(len(NAIVE_TO_PROCESS)):
        axp = [results[f"area_map_{num}_{LABELS[1]}"][i] for num in ITER_NUMS]
        vxp = [results[f"velocity_map_{num}_{LABELS[1]}"][i] for num in ITER_NUMS]
        ncombos.append((([area_map[0]] + axp),
                       ([velocity_map[0]] + vxp)))
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
    qs1 = ax[0, 0].contour(areax, velocityx, q0, 10)
    qs2 = ax[1, 0].contour(areax, velocityx, q0, 10)
    ax[0, 0].clabel(qs1, inline=1, fontsize=10)
    ax[1, 0].clabel(qs2, inline=1, fontsize=10)
    for pair in pcombos:
        ax[0, 0].plot(pair[0], pair[1])
    for pair in ncombos:
        ax[1, 0].plot(pair[0], pair[1])
    ax[0, 0].scatter(area_map[0], velocity_map[0], color="red")
    ax[0, 0].scatter(env.a0, env.v0, color="green")
    ax[0, 0].set_xlabel("Area")
    ax[0, 0].set_ylabel("Velocity")
    ax[0, 0].set_title("Heat Flux")
    ax[1, 0].scatter(area_map[0], velocity_map[0], color="red")
    ax[1, 0].scatter(env.a0, env.v0, color="green")
    ax[1, 0].set_xlabel("Area")
    ax[1, 0].set_ylabel("Velocity")
    ax[1, 0].set_title("Heat Flux")

    ms1 = ax[0, 1].contour(areax, velocityx, m0, 10)
    ms2 = ax[1, 1].contour(areax, velocityx, m0, 10)
    ax[0, 1].clabel(ms1, inline=1, fontsize=10)
    ax[1, 1].clabel(ms2, inline=1, fontsize=10)
    for pair in pcombos:
        ax[0, 1].plot(pair[0], pair[1])
    for pair in ncombos:
        ax[1, 1].plot(pair[0], pair[1])
    ax[0, 1].scatter(area_map[0], velocity_map[0], color="red")
    ax[0, 1].scatter(env.a0, env.v0, color="green")
    ax[0, 1].set_xlabel("Area")
    ax[0, 1].set_ylabel("Velocity")
    ax[0, 1].set_title("Momentum Flux")
    ax[1, 1].scatter(area_map[0], velocity_map[0], color="red")
    ax[1, 1].scatter(env.a0, env.v0, color="green")
    ax[1, 1].set_xlabel("Area")
    ax[1, 1].set_ylabel("Velocity")
    ax[1, 1].set_title("Momentum Flux")

    bs1 = ax[0, 2].contour(areax, velocityx, f0, 10)
    bs2 = ax[1, 2].contour(areax, velocityx, f0, 10)
    ax[0, 2].clabel(bs1, inline=1, fontsize=10)
    ax[1, 2].clabel(bs2, inline=1, fontsize=10)
    for pair in pcombos:
        ax[0, 2].plot(pair[0], pair[1])
    for pair in ncombos:
        ax[1, 2].plot(pair[0], pair[1])
    ax[0, 2].scatter(area_map[0], velocity_map[0], color="red")
    ax[0, 2].scatter(env.a0, env.v0, color="green")
    ax[0, 2].set_xlabel("Area")
    ax[0, 2].set_ylabel("Velocity")
    ax[0, 2].set_title("Buoyancy Flux")
    ax[1, 2].scatter(area_map[0], velocity_map[0], color="red")
    ax[1, 2].scatter(env.a0, env.v0, color="green")
    ax[1, 2].set_xlabel("Area")
    ax[1, 2].set_ylabel("Velocity")
    ax[1, 2].set_title("Buoyancy Flux")
    
    plt.show()
    fig.savefig(os.path.join(SAVE_DIRECTORY, f"flux.svg"))
    plt.close()

    # Save the results
    filepath = os.path.join(SAVE_DIRECTORY, f"results.json")
    j_fp = open(filepath, 'w')
    json.dump(results, j_fp)
    j_fp.close()
