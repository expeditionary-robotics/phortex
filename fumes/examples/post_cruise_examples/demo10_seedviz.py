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


def make_violinplot(numrows, data, labels, ylabel, savename, with_hline=None, lims=None):
    fig, ax = plt.subplots(numrows, 1, sharey=True)
    for i in range(numrows):
        ax[i].violinplot(data[i],
                         showmeans=False,
                         showmedians=True)
        set_axis_style(ax[i], labels[i])
        ax[i].set_ylabel(ylabel)
        if with_hline is not None:
            ax[i].hlines([with_hline], [0], [len(labels[i]) + 1], color="red")
        if lims is not None:
            ax[i].set_ylim(lims)
    plt.savefig(os.path.join(SAVE_DIRECTORY, f"{savename}.svg"))
    plt.close()


SAVE_DIRECTORY = os.path.join(os.getenv("FUMES_OUTPUT"), f"simulations/paperplots_150100")
if not os.path.exists(SAVE_DIRECTORY):
    os.makedirs(SAVE_DIRECTORY)
EXP_PREFIX = ["cloud_map100_iterativeplans_seed",
              "cloud_map150_iterativeplans_seed"]
EXPS_TO_PROCESS = [["260"],  # "227", "244", "541", "648", "670", "809", "986", "658", "572"],  # 100
                   #    ["37", "134", "378", "447", "693", "780", "884", "938", "950", "32"],  # 120
                   ["45"]]  # , "52", "279", "336", "382", "450", "575", "610", "699", "702"]]  # 150
LABELS = ["100m", "150m"]
ITER_NUMS = [0]
PLOT_MODELS = True

# set number of samples for prediction generation
NUM_FORECAST_SAMPS = 10
RANDOM_SEED = 19
np.random.seed(RANDOM_SEED)

# set the z-axis for computing error
QUERYZ = [100., 125., 150.]

# set the horizontal resolution for computing error
EXTENT = Extent(xrange=(-100., 500.),
                xres=100,
                yrange=(-100., 500.),
                yres=100,
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
    # get the environment (since the same, just need to do this once)
    exemplar_params = load_experiment_json(EXP_PREFIX[0] + EXPS_TO_PROCESS[0][0], ITER_NUMS[0])
    times = np.asarray(exemplar_params["simulation_params"]["times"])
    snap_times = [0, 9] #np.unique(np.round(times / 3600.))
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
    
    bas = CrossflowMTT(extent=EXTENT,
                       plume_loc=env_params["plume_loc"],
                       s=env_params["s"],
                       curfunc=curfunc,
                       headfunc=headfunc,
                       tprof=pacific_sp_T,
                       sprof=pacific_sp_S,
                       rhoprof=eos_rho,
                       vex=v0_baseline_samps[4],
                       area=a0_baseline_samps[4],
                       density=env_params["density"],
                       salt=env_params["salt"],
                       temp=env_params["temp"],
                       lam=env_params["lam"],
                       entrainment=(alph_baseline_samps[4], bet_baseline_samps[4]))

    # plot the environment
    if PLOT_MODELS is True:
        env100_snaps = []
        env150_snaps = []
        env_envelopes = []
        bas100_snaps = []
        bas150_snaps = []
        bas_envelopes = []
        for t in snap_times:
            continuous = env.get_snapshot(t=t * 3600, z=[100., 150.])
            env100_snaps.append((continuous[0] > 1e-5).astype(float))
            env150_snaps.append((continuous[1] > 1e-5).astype(float))
            env_envelopes.append(env.envelope(t=t * 3600))

            plt.imshow(continuous[0],
                       cmap="copper",
                       vmin=0,
                       vmax=5e-5,
                       extent=(-100, 500, -100, 500),
                       origin="lower")
            plt.savefig(os.path.join(SAVE_DIRECTORY, f"env_100_{t}.svg"))
            plt.close()

            plt.imshow(continuous[1],
                       cmap="copper",
                       vmin=0,
                       vmax=5e-5,
                       extent=(-100, 500, -100, 500),
                       origin="lower")
            plt.savefig(os.path.join(SAVE_DIRECTORY, f"env_150_{t}.svg"))
            plt.close()

            lee, cle, ree = env_envelopes[-1]
            plt.plot(*cle, label="Centerline", c='g', ls='--')
            plt.plot(*lee, label="Left Extent", c='g')
            plt.plot(*ree, label="Right Extent", c='g')
            lefill = np.interp(ree[0], lee[0], lee[1])
            plt.fill_between(ree[0], ree[1], lefill, interpolate=True, alpha=0.1)
            plt.title("Plume Envelope")
            plt.xlabel("X in crossflow direction (meters)")
            plt.ylabel("Z (meters)")
            plt.axis([-20, 400, -20, 360])
            plt.savefig(os.path.join(SAVE_DIRECTORY, f"env_slice_{t}.svg"))
            plt.close()


            continuous = bas.get_snapshot(t=t * 3600, z=[100., 150.])
            bas100_snaps.append((continuous[0] > 1e-5).astype(float))
            bas150_snaps.append((continuous[1] > 1e-5).astype(float))
            bas_envelopes.append(bas.envelope(t=t * 3600))

            plt.imshow(continuous[0],
                       cmap="copper",
                       vmin=0,
                       vmax=5e-5,
                       extent=(-100, 500, -100, 500),
                       origin="lower")
            plt.savefig(os.path.join(SAVE_DIRECTORY, f"bas_100_{t}.svg"))
            plt.close()

            plt.imshow(continuous[1],
                       cmap="copper",
                       vmin=0,
                       vmax=5e-5,
                       extent=(-100, 500, -100, 500),
                       origin="lower")
            plt.savefig(os.path.join(SAVE_DIRECTORY, f"bas_150_{t}.svg"))
            plt.close()

            lee, cle, ree = env_envelopes[-1]
            plt.plot(*cle, label="Centerline", c='g', ls='--')
            plt.plot(*lee, label="Left Extent", c='g')
            plt.plot(*ree, label="Right Extent", c='g')
            lefill = np.interp(ree[0], lee[0], lee[1])
            plt.fill_between(ree[0], ree[1], lefill, interpolate=True, alpha=0.1)
            plt.title("Plume Envelope")
            plt.xlabel("X in crossflow direction (meters)")
            plt.ylabel("Z (meters)")
            plt.axis([-20, 400, -20, 360])
            plt.savefig(os.path.join(SAVE_DIRECTORY, f"bas_slice_{t}.svg"))
            plt.close()

    for iter in ITER_NUMS:
        for pref, exps, lab in zip(EXP_PREFIX, EXPS_TO_PROCESS, LABELS):
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
                if PLOT_MODELS is True:
                    mod100_snaps = []
                    mod150_snaps = []
                    mod_envelopes = []
                    ious100 = []
                    ious150 = []
                    for i, t in enumerate(snap_times):
                        continuous = mod.get_snapshot(t=t * 3600, z=[100., 150.])
                        mod100_snaps.append(continuous[0] > 1e-5)
                        mod150_snaps.append(continuous[1] > 1e-5)
                        mod_envelopes.append(mod.envelope(t=t * 3600))
                        ious100.append((continuous[0] > 1e-5).astype(float) + env100_snaps[i] * 2)
                        ious150.append((continuous[1] > 1e-5).astype(float) + env150_snaps[i] * 2)

                        plt.imshow(continuous[0],
                                   cmap="copper",
                                   vmin=0,
                                   vmax=5e-5,
                                   extent=(-100, 500, -100, 500),
                                   origin="lower")
                        plt.savefig(os.path.join(SAVE_DIRECTORY, f"mod{exp}_100_{t}.svg"))
                        plt.close()

                        plt.imshow(continuous[1],
                                   cmap="copper",
                                   vmin=0,
                                   vmax=5e-5,
                                   extent=(-100, 500, -100, 500),
                                   origin="lower")
                        plt.savefig(os.path.join(SAVE_DIRECTORY, f"mod{exp}_150_{t}.svg"))
                        plt.close()

                        plt.imshow(ious100[i],
                                   cmap="gnuplot",
                                   origin="lower",
                                   extent=(-100, 500, -100, 500))
                        plt.colorbar()
                        plt.savefig(os.path.join(SAVE_DIRECTORY, f"iou{exp}_100_{t}.svg"))
                        plt.close()

                        plt.imshow(ious150[i],
                                   cmap="gnuplot",
                                   origin="lower",
                                   extent=(-100, 500, -100, 500))
                        plt.colorbar()
                        plt.savefig(os.path.join(SAVE_DIRECTORY, f"iou{exp}_150_{t}.svg"))
                        plt.close()

                        le, cl, re = mod_envelopes[i]
                        lee, cle, ree = env_envelopes[i]
                        leb, clb, reb = bas_envelopes[i]
                        plt.plot(*cl, label="Centerline", c='b', ls='--')
                        plt.plot(*le, label="Left Extent", c='b')
                        plt.plot(*re, label="Right Extent", c='b')
                        lefill = np.interp(re[0], le[0], le[1])
                        plt.fill_between(re[0], re[1], lefill, interpolate=True, alpha=0.1, color='b')
                        # plt.plot(*cle, label="Centerline", c='g', ls='--')
                        # plt.plot(*lee, label="Left Extent", c='g')
                        # plt.plot(*ree, label="Right Extent", c='g')
                        # lefill = np.interp(ree[0], lee[0], lee[1])
                        # plt.fill_between(ree[0], ree[1], lefill, interpolate=True, alpha=0.1, color='g')
                        plt.plot(*clb, label="Centerline", c='r', ls='--')
                        plt.plot(*leb, label="Left Extent", c='r')
                        plt.plot(*reb, label="Right Extent", c='r')
                        lefill = np.interp(reb[0], leb[0], leb[1])
                        plt.fill_between(reb[0], reb[1], lefill, interpolate=True, alpha=0.1, color='r')
                        plt.fill_between(reb[0], reb[1], lefill, interpolate=True, alpha=0.1, color='r')
                        plt.hlines([100, 150], xmin=[0, 0], xmax=[300, 300], colors=['r', 'r'])
                        plt.title("Plume Envelope")
                        plt.xlabel("X in crossflow direction (meters)")
                        plt.ylabel("Z (meters)")
                        plt.axis([-20, 400, -20, 360])
                        plt.savefig(os.path.join(SAVE_DIRECTORY, f"mod{exp}_slice_{t}.svg"))
                        plt.close()

                # get simulated observations
                # sim_obs = np.asarray(exp_json["simulation_params"]["obs"])
                # sim_coords = exp_json["simulation_params"]["coords"]
                # plt.scatter([c[0] for c in sim_coords], [c[1]
                #             for c in sim_coords], c=(sim_obs > 1e-5), cmap="copper", s=0.1)
                # plt.scatter(0, 0, c="red", marker="*")
                # plt.gca().axis("equal")
                # plt.gca().set(xlim=(-100, 600), ylim=(-100, 600))
                # plt.savefig(os.path.join(SAVE_DIRECTORY, f"mod{exp}_coord_{iter}.svg"))
                # plt.close()

                # # get samples of the params
                # area_samps = mod_params["model_learned_params"]["area_samples"]
                # velocity_samps = mod_params["model_learned_params"]["velocity_samples"]
                # plt.hist2d(np.asarray(area_samps).flatten(),
                #            np.asarray(velocity_samps).flatten(),
                #            bins=[20, 20])
                # plt.xlabel("Area")
                # plt.ylabel("Velocity")
                # plt.colorbar()
                # plt.show()
