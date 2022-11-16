"""Creates a Morton-Taylor-Turner model.

This is the parent class for a MTT model and an approxiamte bent-plume version
of this model. This is specifically designed as a model (not environment); on
the cruise we would want to instantiate one of these models so that we can
update parameters from observations then use the model to form predictions.
"""
from os import EX_DATAERR
from matplotlib.artist import allow_rasterization
import numpy as np
from numpy.ma.core import multiply
from sklearn.neighbors import KernelDensity
import copy
import json
import os
import scipy as sp
from distfit import distfit
import warnings
from netCDF4 import Dataset
import gpytorch as gpy
import matplotlib.pyplot as plt
import torch


from fumes.utils import data_home, output_home, tic, toc
from fumes.environment.utils import random_str, ncdump
from fumes.environment.mtt import StationaryMTT, CrossflowMTT, Multiplume

from .model import ScienceModel


class MTT(ScienceModel):
    """Instantiates a single, stationary plume model."""

    def __init__(self, extent, plume_loc, z, tprof, sprof, rhoprof, vex=0.1, area=0.1,
                 density=1000, salt=34.608, temp=300, E=0.255,
                 dive="simulated", experiment_name="temp"):
        """Initializes an MTT model class."""
        self.NAME = f"StationaryModel{random_str()}_{dive}"
        self.experiment_name = experiment_name

        self.extent = extent  # range and resolution of model extent
        self.loc = plume_loc  # (easting, northing, height) plume source coords
        self.z = z  # heights to integrate over in m
        self.tprof_mod = tprof  # GP model of profile
        self.tprof = tprof.profile  # background temperature profile function
        self.sprof_mod = sprof  # GP model of profile
        self.sprof = sprof.profile  # background salinity profile function
        self.rhoprof = rhoprof  # background density profile function
        self.g = -9.81  # acceleration due to gravity N/kg

        # fixed source inputs
        self.rho0 = density  # intial density of the plume in kg/m^3
        self.s0 = salt  # absolute salinity at source in % (unitless)
        self.t0 = temp  # potential temperature at source in C

        # potentially uncertain source inputs
        self.v0 = vex  # initial plume exit velocity
        self.a0 = area  # area of plume orifice

        # physical constants (unknown)
        self.entrainment = E  # entrainment coefficient

        # initialize odesystem with initial conditions
        self.odesys = StationaryMTT(extent=self.extent,
                                    plume_loc=self.loc,
                                    z=z,
                                    tprof=tprof.profile,
                                    sprof=sprof.profile,
                                    rhoprof=rhoprof,
                                    vex=np.mean(vex.sample(5000)),
                                    area=np.mean(area.sample(5000)),
                                    salt=self.s0,
                                    temp=self.t0,
                                    density=self.rho0,
                                    entrainment=np.mean(E.sample(5000)))

        # Ensure the model and underlying odesys have matching names
        self.odesys.NAME = self.NAME

        # metadata objects
        self.prediction_num_samps = None
        self.update_num_samps = None
        self.update_burnin = None
        self.update_thresh = None
        self.used_grid = None

    def set_name(self, name):
        """Overwrites the default model name with input."""
        self.NAME = name
        self.odesys.NAME = name

    def get_parameters(self):
        """Returns the parameters defining the model."""
        print("Exit Velocity: ", np.mean(self.v0.sample(5000)))
        print("Orifice Area: ", np.mean(self.a0.sample(5000)))
        print("Source Temp (C) and Salinity: ", self.t0, self.s0)
        print("Source Density: ", self.rho0)
        print("Entrainment coeff: ", np.mean(self.entrainment.sample(5000)))
        print("Gravity :", self.g)
        return (self.v0, self.a0, self.t0, self.s0, self.rho0,
                self.entrainment, self.g)

    def set_extent_with_z(self, xrange, xres, yrange, yres, z):
        if xrange is None:
            xrange = self.extent.xrange
        if xres is None:
            xres = self.extent.xres
        if yrange is None:
            yrange = self.extent.yrange
        if yres is None:
            yres = self.extent.yres
        if z is None:
            z = np.linspace(self.extent.zrange[0],
                            self.extent.zrange[1],
                            self.extent.zres)
        return xrange, xres, yrange, yres, z

    def set_extent(self, xrange, xres, yrange, yres, zrange, zres):
        if xrange is None:
            xrange = self.extent.xrange
        if xres is None:
            xres = self.extent.xres
        if yrange is None:
            yrange = self.extent.yrange
        if yres is None:
            yres = self.extent.yres
        if zrange is None:
            zrange = self.extent.zrange
        if zres is None:
            zres = self.extent.zres
        return xrange, xres, yrange, yres, zrange, zres

    def _json_stats(self):
        """Creates a dict object about model information for JSON.

        Args:
            z (array of floats): heights to save
            t (array of floats): times to save

        Returns:
            dict object compatible for JSON saving
        """
        json_config_dict = {"model_fixed_params":
                            {"model_type": "stationary",
                             "model_name": self.NAME,
                             "plume_loc": self.loc,
                             "extent": self.extent.get_attributes(),
                             "temp": self.t0,
                             "salt": self.s0,
                             "density": self.rho0,
                             "z": self.z.tolist(),
                             "tprof": self.tprof_mod.get_attributes(),
                             "sprof": self.sprof_mod.get_attributes(),
                             },
                            "model_learned_params":
                            {"velocity_mle": np.mean(self.v0.sample(5000)),
                             "velocity_distribution": self.v0.get_attributes(),
                             "area_mle": np.mean(self.a0.sample(5000)),
                             "area_distribution": self.a0.get_attributes(),
                             "entrainment_mle": np.mean(self.entrainment.sample(5000)),
                             "entrainment_distribution": self.entrainment.get_attributes(),
                             },
                            "model_update_procedure":
                            {"update_samples": self.update_num_samps,
                             "update_burnin": self.update_burnin,
                             "update_thresh": self.update_thresh,
                             "used_grid": self.used_grid,
                             },
                            "model_prediction_procedure":
                            {"prediction_samples": self.prediction_num_samps,
                             },
                            }
        return json_config_dict

    def save_model_metadata(self, overwrite=False, from_multi=None):
        """Creates a JSON file with metadata of the model.

        Args:
            overwrite (bool): whether to overwrite existing file
            from_multi (None or string): pass multiplume name into string

        Returns:
            saves JSON file to disk.
        """
        json_config_dict = self._json_stats()
        if from_multi is not None:
            json_output_file = os.path.join(output_home(),
                                            f"{self.NAME}_multipart_{from_multi}.json")
        else:
            json_output_file = os.path.join(output_home(), f"{self.NAME}.json")
        # check if filename exists, if so you might be overwriting
        if os.path.exists(json_output_file) and overwrite is False:
            warnings.warn("Filename exists and overwrite is False; not saving")
        else:
            j_fp = open(json_output_file, 'w')
            json.dump(json_config_dict, j_fp)
            j_fp.close()

    def solve(self, t, overwrite=True):
        """Pass any updateable params to odesystem.

        Args:
            t (float): timestamp to update
            overwrite (bool): whether to update in-memory cache
        """
        self.odesys.v0 = np.mean(self.v0.sample(5000))
        self.odesys.a0 = np.mean(self.a0.sample(5000))
        self.odesys.entrainment = np.mean(self.entrainment.sample(5000))
        # Note that anytime this is called, it is due to a
        # complete model update. Unless circumstances are different,
        # this call should always overwrite.
        self.odesys.solve(t, overwrite=overwrite)

    def read_cache(self, verbose=False):
        """Given a model cache, read the data.

        Args
            verbose (bool): whether to print ncinfo

        Returns
            pc, xc, yc, zc, tc values from model snapshots
        """
        return self.odesys.read_cache(verbose)

    def read_gp_cache(self, verbose=False, overwrite=True):
        """Given a model cache, read the data.

        Args
            verbose (bool): whether to print ncinfo

        Returns
            pc, xc, yc, zc, tc values from model snapshots
        """
        return self.odesys.read_gp_cache(verbose, overwrite)

    def write_cache(self, tvec, xrange=None, yrange=None, zrange=None,
                    xres=None, yres=None, zres=None, overwrite=True):
        """Updates the environment cache.

        Args:
            tvec (array[float]): time vector,
            xrange (tuple[float]): min/max xdim
            yrange (tuple[float]): min/max ydim
            zrange (tuple[float]): min/max zdim
            xres (int): xdim resolution
            yres (int): ydim resolution
            zres (int): zdim resolution
            overwrite (bool): if True, will overwrite existing cache.
        """
        self.odesys.write_cache(tvec=tvec,
                                xrange=xrange,
                                yrange=yrange,
                                zrange=zrange,
                                xres=xres,
                                yres=yres,
                                zres=zres,
                                overwrite=overwrite)

    def write_gp_cache(self, tvec, xrange=None, yrange=None, zrange=None,
                       xres=None, yres=None, zres=None, overwrite=True,
                       seed_kernel=None, visualize=False):
        """Updates the environment cache.

        Args:
            tvec (array[float]): time vector,
            xrange (tuple[float]): min/max xdim
            yrange (tuple[float]): min/max ydim
            zrange (tuple[float]): min/max zdim
            xres (int): xdim resolution
            yres (int): ydim resolution
            zres (int): zdim resolution
            overwrite (bool): if True, will overwrite existing cache.
            seed_kernel (str): if not None, will seed the kernel of the
                written GP with the contents of the seed_kernel filename
        """
        self.odesys.write_gp_cache(tvec=tvec,
                                   xrange=xrange,
                                   yrange=yrange,
                                   zrange=zrange,
                                   xres=xres,
                                   yres=yres,
                                   zres=zres,
                                   overwrite=overwrite,
                                   seed_kernel=seed_kernel,
                                   visualize=visualize)

    def write_prediction_cache(self, tvec, xrange=(-500., 500.),
                               yrange=(-500., 500.), zrange=(0., 200.),
                               xres=1000, yres=1000, zres=10, overwrite=True,
                               num_samples=100):
        """Updates the model prediction cache.

        Args:
            tvec (array[float]): time vector,
            xrange (tuple[float]): min/max xdim
            yrange (tuple[float]): min/max ydim
            zrange (tuple[float]): min/max zdim
            xres (int): xdim resolution
            yres (int): ydim resolution
            zres (int): zdim resolution
            overwrite (bool): if True, will overwrite existing cache
            num_samples (int): number of posterior samples to use
        """
        self.prediction_cache_name = f"{self.NAME}_predictions_" \
            f"t{tvec[0]}-{tvec[-1]}_" \
            f"x{xrange[0]}-{xrange[1]}_" \
            f"xres{xres}_" \
            f"y{yrange[0]}-{yrange[1]}_" \
            f"yres{yres}_" \
            f"z{zrange[0]}-{zrange[1]}_" \
            f"zres{zres}.nc"
        self.prediction_cache_name = os.path.join(output_home(),
                                                  self.prediction_cache_name)

        # Check if cache already exists
        if os.path.exists(self.prediction_cache_name) and overwrite:
            warnings.warn("Cache already exists. Overwriting ...")
            os.remove(self.prediction_cache_name)
        if os.path.exists(self.prediction_cache_name) and not overwrite:
            warnings.warn("Cache already exists. Returning without writing.")
            return

        ncfile = Dataset(self.prediction_cache_name, "w", format="NETCDF4")
        ncfile.createDimension("time", None)
        ncfile.createDimension("x", xres)
        ncfile.createDimension("y", yres)
        ncfile.createDimension("z", zres)
        ncfile.createDimension("num_samps", 1)

        # Create variables to save data to
        times = ncfile.createVariable("time", "f8", ("time",))
        xs = ncfile.createVariable("x", "f8", ("x",))
        ys = ncfile.createVariable("y", "f8", ("y",))
        zs = ncfile.createVariable("z", "f8", ("z",))
        ns = ncfile.createVariable("num_samps", "f8", ("num_samps",))
        vals = ncfile.createVariable("mean", "f8", ("time", "x", "y", "z",))
        vars = ncfile.createVariable("var", "f8", ("time", "x", "y", "z",))

        # Populate array indices
        xs[:] = np.linspace(xrange[0], xrange[1], xres)  # save x coords
        ys[:] = np.linspace(yrange[0], yrange[1], yres)  # save y coords
        zs[:] = np.linspace(zrange[0], zrange[1], zres)  # save z coords
        times[:] = np.asarray(tvec)  # save time coords
        ns[:] = num_samples  # save number of samples used

        # Populate data
        for i, t in enumerate(tvec):
            tic()
            ms, vs = self.get_snapshot_prediction(
                t=t,
                z=np.linspace(zrange[0], zrange[1], zres),
                xrange=xrange,
                yrange=yrange,
                xres=xres,
                yres=yres,
                from_cache=False,
                num_samples=num_samples)

            for j, zm in enumerate(np.linspace(zrange[0], zrange[1], zres)):
                vals[i, :, :, j] = ms[j, :, :].reshape(xres, yres)
                vars[i, :, :, j] = vs[j, :, :].reshape(xres, yres)
            toc()

        # Close the ncfile
        ncfile.close()

    def read_prediction_cache(self, verbose=False):
        """Given a model cache, read the data.

        Args
            verbose (bool): whether to print ncinfo

        Returns
            pc, vc, xc, yc, zc, tc, nc values from model snapshots
        """
        # Check that the current cache is populated
        assert(self.prediction_cache_name is not None)
        ncfile = Dataset(self.prediction_cache_name, "r")
        if verbose:
            ncdump(ncfile)

        pc = ncfile.variables['mean'][:]  # data
        vc = ncfile.variables['var'][:]  # variance
        xc = ncfile.variables['x'][:]  # xcoords
        yc = ncfile.variables['y'][:]  # ycoords
        zc = ncfile.variables['z'][:]  # zcoords
        tc = ncfile.variables['time'][:]  # tcoords
        nc = ncfile.variables['num_samps'][:]  # number of sample

        # Close the ncfile
        ncfile.close()

        return pc, vc, xc, yc, zc, tc, nc

    def get_value(self, t, loc, from_cache=False, cache_interp="lookup"):
        """Get a deterministic prediction of the model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]) or list(tuple[float]): a location or
                list of locations, in xyz
            from_cache (bool): if True, return nearest value from cache

        Returns: tuple(float) plume probability
        """
        P = self.odesys.get_value(t, loc, return_all=False,
                                  from_cache=from_cache, cache_interp=cache_interp)
        return P

    def get_snapshot(self, t, z=None, xrange=None, yrange=None, xres=None,
                     yres=None, from_cache=False):
        """Get a deterministic prediction of full state at time t.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            from_cache (bool): if True, return nearest value from cache

        Returns: P
        """
        # only want probability of being in plume
        P = self.odesys.get_snapshot(t, z, xrange, yrange, xres, yres,
                                     from_cache=from_cache)
        return P

    def get_maxima(self, t, z=None, xrange=None, yrange=None, xres=None,
                   yres=None, from_cache=False):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            from_cache (bool): if True, return nearest value from cache

        Returns: list[float tuple] maximum location
        """
        # Get snapshot
        loc = self.odesys.get_maxima(t, z, xrange, yrange, xres, yres,
                                     from_cache=from_cache)
        return loc

    def _set_model_parameters_from_sample(self, t, enviro, sid, samples):
        """Sets model parameters from a given sample.

        Args:
            t (float): time index
            sid (int): index of sample
            samples (list of tuples[float]): parameter samples
            enviro (Environment): environment to update

        Returns:
            updates model parameters
        """
        enviro.v0 = samples[0][sid]
        enviro.a0 = samples[1][sid]
        enviro.entrainment = samples[2][sid]
        tic()
        print('here')
        enviro.solve(t=t, overwrite=True)  # always update in sample model
        toc()
        return enviro

    def _sample_param_posterior(self, num_samples=100):
        """Return samples from unknowns."""
        v0 = self.v0.sample(num_samples)
        a0 = self.a0.sample(num_samples)
        En = self.entrainment.sample(num_samples)
        return v0, a0, En

    def get_prediction(self, t, loc, from_cache=False, num_samples=500):
        """Get a (mean, variance) prediction of the model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            from_cache (bool): if True, return nearest value from cache
            num_samples (int): number of posterior samples to draw

        Returns: mean, variance at the location and time
        """
        if from_cache:
            # Read from cache and return nearest values in xyzt
            ms, vs, xs, ys, zs, ts, ns = self.read_prediction_cache()
            qx, qy, qz = loc
            mean = np.zeros_like(qx)
            var = np.zeros_like(qx)

            # Get nearest t in snapshots
            tq = np.fabs(ts - t)
            idt = np.argmin(tq)
            # Get nearest z in snapshots
            zq = np.fabs(zs - qz)
            idz = np.argmin(zq)
            for i, (lx, ly) in enumerate(zip(qx, qy)):
                # Get nearest x in snapshots
                xq = np.fabs(xs - lx)
                idx = np.argmin(xq)
                # Get nearest y in snapshots
                yq = np.fabs(ys - ly)
                idy = np.argmin(yq)
                # Now get mean and variance
                mean[i] = ms[idt, idx, idy, idz]
                var[i] = vs[idt, idx, idy, idz]
            return mean, var

        # If not reading from cache, compute and return
        self.prediction_num_samps = num_samples
        # Sample from parameter values
        temp = self._sample_param_posterior(num_samples=num_samples)
        # Perform operations within a clean model environment
        enviro = copy.deepcopy(self.odesys)
        enviro._model = {}
        vals = []
        # Push samples through get value
        for i in range(0, num_samples):
            enviro = self._set_model_parameters_from_sample(t, enviro, i, temp)
            vals.append(enviro.get_value(t, loc))
        return np.nanmean(vals, axis=0), np.nanvar(vals, axis=0)

    def get_snapshot_prediction(self, t, z=None, xrange=None, yrange=None,
                                xres=None, yres=None, from_cache=False,
                                num_samples=500):
        """Get a deterministic prediction of full state and uncertainty at t.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            from_cache (bool): if True, return nearest value from cache
            num_samples (int): number of posterior samples to use

        Returns: mean and variance across snapshots
        """
        if from_cache:
            # Read from cache and return nearest values in xyzt
            ms, vs, xs, ys, zs, ts, ns = self.read_prediction_cache()
            mean = np.zeros((len(z), xres, yres))
            var = np.zeros((len(z), xres, yres))

            # Get nearest t in snapshots
            tq = np.fabs(ts - t)
            idt = np.argmin(tq)
            for i, zi in enumerate(z):
                # Get nearest z in snapshots
                zq = np.fabs(zs - zi)
                idz = np.argmin(zq)

                # Now have the mean and variance
                mean[i, :, :] = ms[idt, :, :, idz]
                var[i, :, :] = vs[idt, :, :, idz]
            return mean, var

        xrange, xres, yrange, yres, z = self.set_extent_with_z(xrange,
                                                               xres,
                                                               yrange,
                                                               yres,
                                                               z)

        # If not returning from cache, compute and return
        self.prediction_num_samps = num_samples
        # Sample from parameter values
        model_samps = self._sample_param_posterior(num_samples=num_samples)
        # Create clean environment to sample in
        enviro = copy.deepcopy(self.odesys)
        enviro._model = {}
        mean_heights = np.zeros((len(z), xres, yres))
        var_heights = np.zeros((len(z), xres, yres))
        # Push samples through get value
        for i in range(0, num_samples):
            enviro = self._set_model_parameters_from_sample(t, enviro, i, model_samps)
            for k, zh in enumerate(z):
                P = enviro.get_snapshot(t,
                                        [zh],
                                        xrange,
                                        yrange,
                                        xres,
                                        yres,
                                        from_cache=from_cache)
                # compute mean and variance in rolling fashion
                mean_old = copy.deepcopy(mean_heights[k, :, :])
                mean_heights[k, :, :] = mean_old + (P - mean_old) / (i + 1)
                var_heights[k, :, :] = var_heights[k, :, :] + \
                    (P - mean_old) * (P - mean_heights[k, :, :]) / (i + 1)
        return mean_heights, var_heights

    def _compute_sample_prob(self, mod, Es, Vs, As, t, loc, obs, thresh=1e-5):
        def _likelihood(xm, xo):
            if xm == 1 and xo == 1:
                return 0.9
            elif xm == 1 and xo == 0:
                return 0.3  # note: changed on the cruise
            elif xm == 0 and xo == 1:
                return 0.1
            elif xm == 0 and xo == 0:
                return 0.7  # note: changed on the cruise
            else:
                return 0.0

        mod.a0 = As
        mod.v0 = Vs
        mod.entrainment = Es
        mod.solve(t=t, overwrite=True)  # always overwrite in sampling

        P = mod.get_value(t, loc)
        detect = np.log(np.ones_like(P)+P) > thresh
        err = [np.log(_likelihood(float(d), o)) for d, o in zip(detect, obs)]
        err = np.nansum(err)
        prior_E = self.entrainment.predict(Es)
        print(prior_E, Es)
        prior_V = self.v0.predict(Vs)
        print(prior_V, Vs)
        prior_A = self.a0.predict(As)
        print(prior_A, As)
        prop_samp_prob = err + np.log(prior_E) + np.log(prior_V) + np.log(prior_A)
        print(prop_samp_prob)
        return prop_samp_prob

    def _model_sample_chain(self, mod, last_samp, t, loc, obs, thresh=1e-5):
        """Sample model with observation at t, loc.

        Args:
            mod (model): target model to sample
            last_samp (tuple): the last sample in the chain
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            obs (tuple[float]): a sensor observation of in or out
            thresh (float): detection threshold
        """
        Es = np.fabs(last_samp[0] + self.entrainment.sample_proposal(1))[0]
        Vs = np.fabs(last_samp[1] + self.v0.sample_proposal(1))[0]
        As = np.fabs(last_samp[2] + self.a0.sample_proposal(1))[0]

        prop_samp_prob = self._compute_sample_prob(mod, Es, Vs, As, t,
                                                   loc, obs, thresh=thresh)
        prop_samp = (Es, Vs, As)

        return prop_samp, prop_samp_prob

    def update(self, t, loc, obs, num_samps=100, burnin=50, thresh=1e-5, use_grid=False):
        """Update model with observation at t, loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            obs (tuple[float]): a sensor observation of in or out
            num_samps (int): number of parameter samples to draw
            burnin (int): number of samples to drop
            thresh (float): detection threshold
            use_grid (bool): whether to perform grid search instead
        """
        self.update_num_samps = num_samps
        self.update_burnin = burnin
        self.update_thresh = thresh
        self.used_grid = use_grid

        # create a place to save model checkpoints
        filepath = os.path.join(output_home(), f"modeling/{self.experiment_name}")
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)

        # create clean environment to perform work in
        enviro = copy.deepcopy(self.odesys)
        enviro._model = {}

        if use_grid is False:
            # draw samples from distributions to update
            Et = np.mean(self.entrainment.sample(num_samps))
            Vt = np.mean(self.v0.sample(num_samps))
            At = np.mean(self.a0.sample(num_samps))

            last_samp = (Et, Vt, At)
            last_samp_prob = self._compute_sample_prob(enviro, Et, Vt, At, t,
                                                       loc, obs, thresh=thresh)
            samples = np.zeros((num_samps, 3))
            samples[0, :] = last_samp
            naccept = 1
            accept_tracker = [naccept]

            for i in range(1, num_samps):
                print("Computing sample ", i)
                if (i + 1) % 10 == 0:
                    plt.plot(range(len(accept_tracker)), accept_tracker)
                    plt.xlabel("Sample Num")
                    plt.ylabel("Number Accepted Samples")
                    plt.title("Accepted Samples in Chain")
                    plt.savefig(os.path.join(filepath, "num_accepted_samples.svg"))
                    plt.close()

                    plt.plot(range(len(samples[:, 0])), samples[:, 0])
                    plt.xlabel("Sample Num")
                    plt.ylabel("Entrainment Sample Values")
                    plt.title("Entrainment Samples")
                    plt.savefig(os.path.join(filepath, "entrainment_samples.svg"))
                    plt.close()

                    plt.plot(np.linspace(0, 1, 100), self.entrainment.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
                    plt.hist(samples[:, 0], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=False)
                    plt.xlabel("Entrainment Sample Values")
                    plt.ylabel("PDF")
                    plt.title("Entrainment Samples")
                    plt.savefig(os.path.join(filepath, "entrainment_distribution.svg"))
                    plt.close()

                    plt.plot(range(len(samples[:, 1])), samples[:, 1])
                    plt.xlabel("Sample Num")
                    plt.ylabel("Velocity Sample Values")
                    plt.title("Exit Velocity Samples")
                    plt.savefig(os.path.join(filepath, "velocity_samples.svg"))
                    plt.close()

                    plt.plot(np.linspace(0, 2, 100), self.v0.predict(np.linspace(0, 2, 100)), linewidth=3, alpha=0.5)
                    plt.hist(samples[:, 1], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=False)
                    plt.xlabel("Velocity Sample Values")
                    plt.ylabel("PDF")
                    plt.title("Velocity Samples")
                    plt.savefig(os.path.join(filepath, "velocity_distribution.svg"))
                    plt.close()

                    plt.plot(range(len(samples[:, 2])), samples[:, 2])
                    plt.xlabel("Sample Num")
                    plt.ylabel("Area Sample Values")
                    plt.title("Vent Area Samples")
                    plt.savefig(os.path.join(filepath, "area_samples.svg"))
                    plt.close()

                    plt.plot(np.linspace(0, 1, 100), self.entrainment.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
                    plt.hist(samples[:, 2], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=False)
                    plt.xlabel("Area Sample Values")
                    plt.ylabel("PDF")
                    plt.title("Area Samples")
                    plt.savefig(os.path.join(filepath, "area_distribution.svg"))
                    plt.close()

                prop_samp, prop_samp_prob = self._model_sample_chain(enviro,
                                                                     last_samp,
                                                                     t,
                                                                     loc,
                                                                     obs,
                                                                     thresh=thresh)
                rho = min(1, np.exp(prop_samp_prob - last_samp_prob))
                u = np.random.uniform()
                if u < rho:
                    naccept += 1
                    last_samp_prob = prop_samp_prob
                    last_samp = prop_samp
                accept_tracker.append(naccept)
                samples[i, :] = last_samp
            print("Number of accepted samples in chain:", naccept)
            plt.plot(range(len(accept_tracker)), accept_tracker)
            plt.xlabel("Sample Num")
            plt.ylabel("Number Accepted Samples")
            plt.title("Accepted Samples in Chain")
            plt.savefig(os.path.join(filepath, "num_accepted_samples.svg"))
            plt.close()

            plt.plot(range(len(samples[:, 0])), samples[:, 0])
            plt.xlabel("Sample Num")
            plt.ylabel("Entrainment Sample Values")
            plt.title("Entrainment Samples")
            plt.savefig(os.path.join(filepath, "entrainment_samples.svg"))
            plt.close()

            plt.plot(np.linspace(0, 1, 100), self.entrainment.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
            plt.hist(samples[:, 0], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=False)
            plt.xlabel("Entrainment Sample Values")
            plt.ylabel("PDF")
            plt.title("Entrainment Samples")
            plt.savefig(os.path.join(filepath, "entrainment_distribution.svg"))
            plt.close()

            plt.plot(range(len(samples[:, 1])), samples[:, 1])
            plt.xlabel("Sample Num")
            plt.ylabel("Velocity Sample Values")
            plt.title("Exit Velocity Samples")
            plt.savefig(os.path.join(filepath, "velocity_samples.svg"))
            plt.close()

            plt.plot(np.linspace(0, 2, 100), self.v0.predict(np.linspace(0, 2, 100)), linewidth=3, alpha=0.5)
            plt.hist(samples[:, 1], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=False)
            plt.xlabel("Velocity Sample Values")
            plt.ylabel("PDF")
            plt.title("Velocity Samples")
            plt.savefig(os.path.join(filepath, "velocity_distribution.svg"))
            plt.close()

            plt.plot(range(len(samples[:, 2])), samples[:, 2])
            plt.xlabel("Sample Num")
            plt.ylabel("Area Sample Values")
            plt.title("Vent Area Samples")
            plt.savefig(os.path.join(filepath, "area_samples.svg"))
            plt.close()

            plt.plot(np.linspace(0, 1, 100), self.entrainment.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
            plt.hist(samples[:, 2], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=False)
            plt.xlabel("Area Sample Values")
            plt.ylabel("PDF")
            plt.title("Area Samples")
            plt.savefig(os.path.join(filepath, "area_distribution.svg"))
            plt.close()
            
            # set new param
            self.entrainment.update(samples[burnin:, 0])
            self.v0.update(samples[burnin:, 1])
            self.a0.update(samples[burnin:, 2])
        else:
            # methodically go through all combinations of samples to pick the best
            Et = self.entrainment.sample(num_samps)
            Vt = self.v0.sample(num_samps)
            At = self.a0.sample(num_samps)

            best_error = 1e10
            best_option = None
            for et in Et:
                for at in At:
                    for vt in Vt:
                        enviro.a0 = at
                        enviro.v0 = vt
                        enviro.entrainment = et
                        enviro.solve(t=t, overwrite=True)  # always overwrite in sampling

                        P = enviro.get_value(t, loc)
                        detect = P > thresh
                        err = np.sum(np.fabs(detect - obs))
                        if err < best_error:
                            best_error = err
                            best_option = (et, vt, at)
            # set new param
            self.entrainment.update(np.random.normal(best_option[0], 0.05, 1000))
            self.v0.update(np.random.normal(best_option[1], 0.1, 1000))
            self.a0.update(np.random.normal(best_option[2], 0.1, 1000))

        self.odesys._model = {}  # Reset environment
        self.solve(t=t, overwrite=True)  # Now update in-memory model
        return np.mean(self.entrainment.sample(5000)), \
            np.mean(self.v0.sample(5000)), np.mean(self.a0.sample(5000))


class Crossflow(MTT):
    """Instantiates a single, crossflow plume model."""

    def __init__(self, extent, plume_loc, s, curfunc, headfunc, tprof, sprof, rhoprof,
                 vex=0.1, area=0.1, density=1000, salt=34.608, temp=300,
                 E=(0.255, 0.0), dive="simulated", experiment_name="temp"):
        """Initializes an MTT model class."""
        self.NAME = f"CrossflowModel{random_str()}_{dive}"
        print(self.NAME)

        self.loc = plume_loc  # (easting, northing, height) plume source coords
        self.extent = extent  # range and resolution of model extent
        self.s = s  # heights to integrate over in m
        self.tprof_mod = tprof
        self.tprof = tprof.profile  # background temperature profile function
        self.sprof_mod = sprof
        self.sprof = sprof.profile  # background salinity profile function
        self.rhoprof = rhoprof  # background density profile function
        self.g = -9.81  # acceleration due to gravity N/kg

        # fixed source inputs
        self.rho0 = density  # intial density of the plume in kg/m^3
        self.s0 = salt  # absolute salinity at source in % (unitless)
        self.t0 = temp  # potential temperature at source in C

        # uncertain enviro inputs
        # self.currents = curfunc.magnitude  # function that describes currents
        # self.heading = headfunc.heading  # function that describes current heading
        self.curr_mag_sampler = curfunc  # sampler for current magnitude
        self.curr_head_sampler = headfunc  # sampler for current heading

        # uncertain source inputs
        self.v0 = vex  # initial plume exit velocity
        self.a0 = area  # area of plume orifice

        # physical constants (unknown)
        self.entrainment = E  # entrainment coefficient

        # initialize odesystem with initial conditions
        self.odesys = CrossflowMTT(extent=self.extent,
                                   plume_loc=self.loc,
                                   s=s,
                                   tprof=tprof.profile,
                                   sprof=sprof.profile,
                                   rhoprof=rhoprof,
                                   curfunc=curfunc.magnitude,
                                   headfunc=headfunc.heading,
                                   vex=np.mean(vex.sample(5000)),
                                   area=np.mean(area.sample(5000)),
                                   salt=self.s0,
                                   temp=self.t0,
                                   density=self.rho0,
                                   entrainment=(np.mean(E[0].sample(5000)),
                                                np.mean(E[1].sample(5000))))
        # Ensure the model and underlying odesys have matching names
        self.odesys.NAME = self.NAME

        # metadata containers
        self.update_num_samps = None
        self.update_burnin = None
        self.update_thresh = None
        self.used_grid = None
        self.prediction_num_samps = None
        self.chain = None

    def get_parameters(self):
        """Returns the parameters defining the model."""
        print("Exit Velocity: ", np.mean(self.v0.sample(5000)))
        print("Orifice Area: ", np.mean(self.a0.sample(5000)))
        print("Source Temp (C) and Salinity: ", self.t0, self.s0)
        print("Source Density: ", self.rho0)
        print("Entrainment coeff: ", np.mean(self.entrainment[0].sample(
            5000)), np.mean(self.entrainment[1].sample(5000)))
        print("Gravity :", self.g)
        return (self.v0, self.a0, self.t0, self.s0, self.rho0,
                self.entrainment, self.g)

    def _json_stats(self, z=None, t=None):
        """Creates a dict object about model information for JSON.

        Args:
            z (array of floats): heights to save
            t (array of floats): times to save

        Returns:
            dict object compatible for JSON saving
        """
        if z is None:
            z = np.linspace(0, 200, 100)

        if t is None:
            t = np.linspace(0, 3600 * 24, 25)
        mode_data = np.asarray([self.v0.sample(5000),
                                self.a0.sample(5000),
                                self.entrainment[0].sample(5000),
                                self.entrainment[1].sample(5000)]).reshape(4, 5000)
        kde = KernelDensity(kernel='gaussian',
                            bandwidth=0.5).fit(mode_data)
        height = np.exp(kde.score_samples(mode_data[:][:]))
        maps = mode_data[:][:][np.argmax(height)]
        json_config_dict = {"model_fixed_params":
                            {"model_type": "crossflow",
                             "model_name": self.NAME,
                             "plume_loc": self.loc,
                             "extent": self.extent.get_attributes(),
                             "temp": self.t0,
                             "salt": self.s0,
                             "density": self.rho0,
                             "s": self.s.tolist(),
                             "z": z.tolist(),
                             "tprof": self.tprof_mod.get_attributes(),
                             "sprof": self.sprof_mod.get_attributes(),
                             "t": t.tolist(),
                             },
                            "model_learned_params":
                            {"velocity_mle": maps[0],
                             "velocity_distribution": self.v0.get_attributes(),
                             "velocity_samples": self.v0.sample(1000).tolist(),
                             "area_mle": maps[1],
                             "area_distribution": self.a0.get_attributes(),
                             "area_samples": self.a0.sample(1000).tolist(),
                             "entrainment_alpha_mle": maps[2],
                             "entrainment_alpha_distribution": self.entrainment[0].get_attributes(),
                             "entrainment_beta_mle": maps[3],
                             "entrainment_beta_distribution": self.entrainment[1].get_attributes(),
                             "curr_magnitude": self.curr_mag_sampler.get_attributes(),
                             "curr_heading": self.curr_head_sampler.get_attributes(),
                             },
                            "model_update_procedure":
                            {"update_samples": self.update_num_samps,
                             "update_burnin": self.update_burnin,
                             "update_thresh": self.update_thresh,
                             "used_grid": self.used_grid,
                             "chain_samples": self.chain.tolist(),
                             },
                            "model_prediction_procedure":
                            {"prediction_samples": self.prediction_num_samps,
                             },
                            }
        return json_config_dict

    def save_model_metadata(self, overwrite=False, from_multi=None):
        """Creates a JSON file with metadata of the model.

        Args:
            overwrite (bool): whether to overwrite existing file
            from_multi (None or string): name of multiplume class

        Returns:
            saves JSON file to disk.
        """
        z = np.linspace(0, 100, 100)
        t = np.linspace(0, 12 * 3600, 24)
        # TODO: change current and heading interface when uncertainty available
        json_config_dict = self._json_stats(z, t)
        if from_multi is not None:
            json_output_file = os.path.join(output_home(),
                                            f"{self.NAME}_multipart_{from_multi}.json")
        else:
            json_output_file = os.path.join(output_home(), f"{self.NAME}.json")
        # check if filename exists, if so you might be overwriting
        if os.path.exists(json_output_file) and overwrite is False:
            warnings.warn("Filename exists and overwrite is False; not saving")
        else:
            j_fp = open(json_output_file, 'w')
            json.dump(json_config_dict, j_fp)
            j_fp.close()

    def solve(self, t, overwrite=True):
        """Pass any updateable params to odesystem.

        Args:
            t (float): global time to set internal models to.
        """
        mode_data = np.asarray([self.v0.sample(5000),
                                self.a0.sample(5000),
                                self.entrainment[0].sample(5000),
                                self.entrainment[1].sample(5000)]).reshape(4, 5000)
        print(mode_data.shape)
        kde = KernelDensity(kernel='gaussian',
                            bandwidth=0.05).fit(mode_data[:][:])
        height = np.exp(kde.score_samples(mode_data[:][:]))
        maps = mode_data[:][:][np.argmax(height)]
        print(maps)
        self.odesys.v0 = maps[0]
        self.odesys.a0 = maps[1]
        self.odesys.entrainment = (maps[2],
                                   maps[3])
        self.odesys.solve(t, overwrite=overwrite)

    def _compute_sample_prob(self, mod, Alphs, Bets, Vs, As, t,
                             loc, obs, thresh=1e-5):
        """Computes a given samples MCMH score.

        Args:
            mod (model): target model to sample
            Alphs (float): sample entrainment alpha coefficient
            Bets (float): sample entrainment beta coefficient
            Vs (float): sample exit velocity
            As (float): sample source area
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            obs (tuple[float]): a sensor observation of in or out
            thresh (float): detection threshold

        Returns: MCMH probability score (float)
        """
        def _likelihood_br(xm, x0):
            if xm == 1:
                return 0.9
            elif xm == 0:
                return 0.3
            else:
                return 0.0

        def _likelihood(xm, xo):
            if xm == 1 and xo == 1:
                return 0.9
            elif xm == 1 and xo == 0:
                return 0.3  # changed on cruise
            elif xm == 0 and xo == 1:
                return 0.1
            elif xm == 0 and xo == 0:
                return 0.7  # changed on cruise
            else:
                return 0.0

        mod.a0 = As
        mod.v0 = Vs
        mod.entrainment = (Alphs, Bets)

        err_br = []
        for i, tt in enumerate(t):
            mod.solve(t=tt, overwrite=True)
            P = mod.get_value(tt, loc[i][:])
            detect = np.log(np.ones_like(P)+P) > thresh
            err_brier = [(_likelihood_br(d, o) - o)**2 for d, o in zip(detect, obs[i][:])]
            err_br = err_br + err_brier
        prior_Alph = self.entrainment[0].predict(Alphs)
        prior_Bet = self.entrainment[1].predict(Bets)
        prior_V = self.v0.predict(Vs)
        prior_A = self.a0.predict(As)
        prop_samp_prob = np.nanmean(err_br) - np.log(prior_Alph) - np.log(prior_Bet) - np.log(prior_V) - np.log(prior_A)
        return prop_samp_prob

    def _model_sample_chain(self, mod, last_samp, t, loc, obs, thresh=1e-5):
        """Sample model with observation at t, loc.

        Args:
            mod (model): target model to sample
            last_samp (tuple): the last sample in the chain
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            obs (tuple[float]): a sensor observation of in or out
            thresh (float): detection threshold

        Returns: sample, sample MCMH score
        """
        Alphs = np.fabs(last_samp[0] +
                        self.entrainment[0].sample_proposal(1))[0]
        Bets = np.fabs(last_samp[1] +
                       self.entrainment[1].sample_proposal(1))[0]
        Vs = np.fabs(last_samp[2] + self.v0.sample_proposal(1))[0]
        As = np.fabs(last_samp[3] + self.a0.sample_proposal(1))[0]

        prop_samp_prob = self._compute_sample_prob(
            mod, Alphs, Bets, Vs, As, t, loc, obs, thresh=thresh)
        prop_samp = (Alphs, Bets, Vs, As)

        return prop_samp, prop_samp_prob

    def update(self, t, loc, obs, num_samps=100, burnin=50, thresh=1e-5, use_grid=False):
        """Update model with observation at t, loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            obs (tuple[float]): a sensor observation of in or out
            num_samps (int): number of parameter samples to draw
            burnin (int): number of samples to drop
            thresh (float): detection threshold
            use_grid (bool): whether to use grid search
            quick_viz (bool): whether to output histograms of samples at end
        """
        self.update_num_samps = num_samps
        self.update_burnin = burnin
        self.update_thresh = thresh
        self.used_grid = use_grid

        # create a place to save model checkpoints
        filepath = os.path.join(output_home(), f"modeling/{self.experiment_name}")
        if not os.path.exists(filepath):
            os.makedirs(filepath, exist_ok=True)

        # create clean world to operate
        enviro = copy.deepcopy(self.odesys)
        enviro._model = {}

        if use_grid is False:
            # draw samples from distributions to update
            Alpht = np.mean(self.entrainment[0].sample(num_samps))
            Bett = np.mean(self.entrainment[1].sample(num_samps))
            Vt = np.mean(self.v0.sample(num_samps))
            At = np.mean(self.a0.sample(num_samps))
            last_samp = (Alpht, Bett, Vt, At)
            last_samp_prob = self._compute_sample_prob(
                enviro, Alpht, Bett, Vt, At, t, loc, obs, thresh=thresh)

            samples = np.zeros((num_samps, 4))
            samples[0, :] = last_samp
            naccept = 1
            accept_tracker = [naccept]

            for i in range(1, num_samps):
                print("Computing Sample: ", i)
                if (i+1) % 10 == 0:
                    plt.plot(range(len(accept_tracker)), accept_tracker)
                    plt.plot(range(len(accept_tracker)), range(len(accept_tracker)), "--")
                    plt.xlabel("Sample Num")
                    plt.ylabel("Number Accepted Samples")
                    plt.title("Accepted Samples in Chain")
                    plt.savefig(os.path.join(filepath, "num_accepted_samples.svg"))
                    plt.close()

                    plt.plot(range(len(samples[:, 0])), samples[:, 0])
                    plt.xlabel("Sample Num")
                    plt.ylabel("Alpha Sample Values")
                    plt.title("Alpha Samples")
                    plt.savefig(os.path.join(filepath, "alpha_samples.svg"))
                    plt.close()

                    plt.plot(range(len(samples[:, 1])), samples[:, 1])
                    plt.xlabel("Sample Num")
                    plt.ylabel("Beta Sample Values")
                    plt.title("Beta Samples")
                    plt.savefig(os.path.join(filepath, "beta_samples.svg"))
                    plt.close()

                    plt.plot(range(len(samples[:, 2])), samples[:, 2])
                    plt.xlabel("Sample Num")
                    plt.ylabel("Velocity Sample Values")
                    plt.title("Exit Velocity Samples")
                    plt.savefig(os.path.join(filepath, "velocity_samples.svg"))
                    plt.close()

                    plt.plot(range(len(samples[:, 3])), samples[:, 3])
                    plt.xlabel("Sample Num")
                    plt.ylabel("Area Sample Values")
                    plt.title("Vent Area Samples")
                    plt.savefig(os.path.join(filepath, "area_samples.svg"))
                    plt.close()

                    if i > burnin:
                        print("Saving model updated params...")
                        save_Alph = copy.deepcopy(self.entrainment[0])
                        save_Bet = copy.deepcopy(self.entrainment[1])
                        save_V = copy.deepcopy(self.v0)
                        save_A = copy.deepcopy(self.a0)

                        save_Alph.update(samples[burnin:, 0])
                        save_Bet.update(samples[burnin:, 1])
                        save_V.update(samples[burnin:, 2])
                        save_A.update(samples[burnin:, 3])

                        v0_mean = np.mean(save_V.sample(5000))
                        a0_mean = np.mean(save_A.sample(5000))
                        alph_mean = np.mean(save_Alph.sample(5000))
                        bet_mean = np.mean(save_Bet.sample(5000))

                        print("Saved V, A, Alph, Bet: ", (v0_mean, a0_mean, alph_mean, bet_mean))

                prop_samp, prop_samp_prob = self._model_sample_chain(
                    enviro, last_samp, t, loc, obs, thresh=thresh)
                if prop_samp_prob > last_samp_prob:
                    rho = min(1, np.exp(last_samp_prob - prop_samp_prob))
                    print(np.exp(last_samp_prob - prop_samp_prob))
                    u = np.random.uniform()
                    if u < rho:
                        naccept += 1
                        last_samp_prob = prop_samp_prob
                        last_samp = prop_samp
                else:
                    naccept += 1
                    last_samp_prob = prop_samp_prob
                    last_samp = prop_samp
                accept_tracker.append(naccept)
                samples[i, :] = last_samp
            
            # set new param
            self.entrainment[0].update(samples[burnin:, 0])
            self.entrainment[1].update(samples[burnin:, 1])
            self.v0.update(samples[burnin:, 2])
            self.a0.update(samples[burnin:, 3])
            self.chain = samples
            
            print("Number of accepted samples in chain:", naccept)
            plt.plot(range(len(accept_tracker)), accept_tracker)
            plt.plot(range(len(accept_tracker)), range(len(accept_tracker)), "--")
            plt.xlabel("Sample Num")
            plt.ylabel("Number Accepted Samples")
            plt.title("Accepted Samples in Chain")
            plt.savefig(os.path.join(filepath, "num_accepted_samples.svg"))
            plt.close()

            plt.plot(range(len(samples[:, 0])), samples[:, 0])
            plt.xlabel("Sample Num")
            plt.ylabel("Alpha Sample Values")
            plt.title("Alpha Samples")
            plt.savefig(os.path.join(filepath, "alpha_samples.svg"))
            plt.close()

            plt.plot(np.linspace(0, 1, 100), self.entrainment[0].predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
            plt.hist(samples[:, 0], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
            plt.xlabel("Alpha Sample Values")
            plt.ylabel("PDF")
            plt.title("Alpha Samples")
            plt.savefig(os.path.join(filepath, "alpha_distribution.svg"))
            plt.close()

            plt.plot(range(len(samples[:, 1])), samples[:, 1])
            plt.xlabel("Sample Num")
            plt.ylabel("Beta Sample Values")
            plt.title("Beta Samples")
            plt.savefig(os.path.join(filepath, "beta_samples.svg"))
            plt.close()

            plt.plot(np.linspace(0, 1, 100), self.entrainment[1].predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
            plt.hist(samples[:, 1], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
            plt.xlabel("Beta Sample Values")
            plt.ylabel("PDF")
            plt.title("Beta Samples")
            plt.savefig(os.path.join(filepath, "beta_distribution.svg"))
            plt.close()

            plt.plot(range(len(samples[:, 2])), samples[:, 2])
            plt.xlabel("Sample Num")
            plt.ylabel("Velocity Sample Values")
            plt.title("Exit Velocity Samples")
            plt.savefig(os.path.join(filepath, "velocity_samples.svg"))
            plt.close()

            plt.plot(np.linspace(0, 2, 100), self.v0.predict(np.linspace(0, 2, 100)), linewidth=3, alpha=0.5)
            plt.hist(samples[:, 2], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
            plt.xlabel("Velocity Sample Values")
            plt.ylabel("PDF")
            plt.title("Velocity Samples")
            plt.savefig(os.path.join(filepath, "velocity_distribution.svg"))
            plt.close()

            plt.plot(range(len(samples[:, 3])), samples[:, 3])
            plt.xlabel("Sample Num")
            plt.ylabel("Area Sample Values")
            plt.title("Vent Area Samples")
            plt.savefig(os.path.join(filepath, "area_samples.svg"))
            plt.close()

            plt.plot(np.linspace(0, 1, 100), self.a0.predict(np.linspace(0, 1, 100)), linewidth=3, alpha=0.5)
            plt.hist(samples[:, 3], 10, fc='gray', histtype='stepfilled', alpha=0.3, density=True)
            plt.xlabel("Area Sample Values")
            plt.ylabel("PDF")
            plt.title("Area Samples")
            plt.savefig(os.path.join(filepath, "area_distribution.svg"))
            plt.close()
            
        else:
            # methodically go through all combinations of samples to pick the best
            Alpht = self.entrainment[0].sample(num_samps)
            Bett = self.entrainment[1].sample(num_samps)
            Vt = self.v0.sample(num_samps)
            At = self.a0.sample(num_samps)

            best_error = 1e10
            best_option = None
            for alpht in Alpht:
                for bett in Bett:
                    for vt in Vt:
                        for at in At:
                            enviro.a0 = at
                            enviro.v0 = vt
                            enviro.entrainment = (alpht, bett)
                            err = 0
                            for i, tt in enumerate(t):
                                enviro.solve(t=tt, overwrite=True)
                                if type(loc) == list:
                                    P = enviro.get_value(tt, loc[i][:])
                                else:
                                    P = enviro.get_value(tt, loc[i, :, :])
                                detect = P > thresh
                                if type(obs) == list:
                                    err += np.sum(np.fabs(detect - obs[i][:]))
                                else:
                                    err += np.sum(np.fabs(detect - obs[i, :]))
                            if err < best_error:
                                best_error = err
                                best_option = (alpht, bett, vt, at)
                print("Current best option (alph, bet, V, A): ", best_option, best_error)
            # set new param
            self.entrainment[0].update(np.random.normal(best_option[0], 0.05, 1000))
            self.entrainment[1].update(np.random.normal(best_option[1], 0.05, 1000))
            self.v0.update(np.random.normal(best_option[2], 0.1, 1000))
            self.a0.update(np.random.normal(best_option[3], 0.1, 1000))

        self.odesys._model = {}  # clean out old environment
        self.solve(t=t[-1], overwrite=True)  # now update environment
        return sp.stats.mode(self.entrainment[0].sample(5000))[0], \
            sp.stats.mode(self.entrainment[1].sample(5000))[0], \
            sp.stats.mode(self.v0.sample(5000))[0], sp.stats.mode(self.a0.sample(5000))[0]

    def _sample_param_posterior(self, num_samples=100):
        """Return samples from unknowns."""
        v0 = self.v0.sample(num_samples)
        a0 = self.a0.sample(num_samples)
        En = (self.entrainment[0].sample(num_samples),
              self.entrainment[1].sample(num_samples))
        curr_mag = self.curr_mag_sampler.sample_magnitude(num_samples)
        curr_head = self.curr_head_sampler.sample_heading(num_samples)
        return v0, a0, En, curr_mag, curr_head

    def _set_model_parameters_from_sample(self, t, enviro, sid, samples):
        """Updates the model from parameters drawn from a sample.

        Args:
            t (float): time index
            enviro (Environment): environment to update
            sid (int): index of sample
            samples (tuple[float]): sample to update params with

        Returns:
            updated model class
        """
        enviro.v0 = samples[0][sid]
        enviro.a0 = samples[1][sid]
        enviro.entrainment = (samples[2][0][sid], samples[2][1][sid])
        enviro.currents = lambda z, t: samples[3][sid](t)
        enviro.heading = lambda t: samples[4][sid](t)
        enviro.solve(t=t, overwrite=True)
        return enviro


class Multimodel(Crossflow):
    """Wrapper for multiple environment models."""

    def __init__(self, multiplume_models, dive="simulated"):
        """Stores a list of plume models."""
        self.NAME = f"Multimodel{random_str()}_{dive}"

        self.models = multiplume_models  # list of models
        ext = self.models[0].extent
        # TODO: turn this back on, but need to check attributes
        # for i in range(1, len(self.models)):
        #     try:
        #         assert(self.models[i].extent == ext)
        #     except:
        #         import pdb; pdb.set_trace()
        # Ensure underlying models have the same name
        self.extent = ext
        # create a multiplume environment from model classes
        self.enviro = Multiplume([m.odesys for m in self.models])

        # Ensure the model and underlying odesys have matching names
        self.enviro.NAME = self.NAME

    def set_name(self, name, model_names):
        """Overwrites the default model name with input.

        Args:
            name (str): the name for the global, multiplume model.
            model_names (list[str]): a list of names of the same length
                as self.models for each consituant model
        """

        self.NAME = name
        self.enviro.NAME = name
        for i, m in enumerate(self.models):
            m.NAME = model_names[i]
        for i, m in enumerate(self.enviro.models):
            m.NAME = model_names[i]

    def solve(self, t, overwrite=True):
        """Pass any updateable params to odesystem.

        Args:
            t (float): time stamp to update in-memory model to
            overwrite (bool): whether to update underlying enviro
        """
        for m in self.models:
            m.solve(t=t, overwrite=overwrite)
        self.enviro = Multiplume([m.odesys for m in self.models])

    def _json_stats(self):
        """Return JSON dicts of model information."""
        model_dicts = []
        for m in self.models:
            model_dicts.append(m._json_stats())
        return model_dicts

    def save_model_metadata(self, overwrite=False):
        """Save each model's metadata to file.

        Args:
            overwrite (bool): whether to overwrite an existing file

        Returns:
            saves files for each model to disk
        """
        for m in self.models:
            m.save_model_metadata(overwrite=overwrite, from_multi=self.NAME)

    def update(self, t, loc, obs, num_samps=100, burnin=50, thresh=1e-5):
        """Update model with observation at t, loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            obs (tuple[float]): a sensor observation of in or out
            num_samps (int): number of parameter samples to draw
            burnin (int): number of samples to drop
            thresh (float): detection threshold
        """
        out = []  # provide an indicator about updated values
        for i, m in enumerate(self.models):
            status = m.update(t,
                              loc,
                              obs,
                              num_samps=num_samps,
                              burnin=burnin,
                              thresh=thresh)
            out.append(status)
        self.enviro = Multiplume([m.odesys for m in self.models])
        return out

    def get_value(self, t, loc, from_cache=False, cache_interp="lookup"):
        """Get a deterministic prediction of the model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]) or list(tuple[float]): a location or
                list of locations, in xyz
            from_cache (bool): if True, return nearest value from cache

        Returns: tuple(float) plume probability
        """
        P = self.enviro.get_value(t, loc, from_cache=from_cache,
                                  cache_interp=cache_interp)
        return P

    def get_prediction(self, t, loc, num_samples=500, from_cache=None):
        """Get model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]): a location, in xyz
            from_cache (string): cache pointer
            num_samples (int): number of posterior samples to use

        Returns: mean and variance of multiplume models
        """
        if from_cache:
            # Read from cache and return nearest values in xyzt
            ms, vs, xs, ys, zs, ts, ns = self.read_prediction_cache()
            qx, qy, qz = loc
            mean = np.zeros_like(qx)
            var = np.zeros_like(qx)

            # Get nearest t in snapshots
            tq = np.fabs(ts - t)
            idt = np.argmin(tq)
            # Get nearest z in snapshots
            zq = np.fabs(zs - qz)
            idz = np.argmin(zq)
            for i, (lx, ly) in enumerate(zip(qx, qy)):
                # Get nearest x in snapshots
                xq = np.fabs(xs - lx)
                idx = np.argmin(xq)
                # Get nearest y in snapshots
                yq = np.fabs(ys - ly)
                idy = np.argmin(yq)
                # Now get mean and variance
                mean[i] = ms[idt, idx, idy, idz]
                var[i] = vs[idt, idx, idy, idz]
            return mean, var

        # If not using cache, compute and return
        vals = []
        envs = []
        # Push samples through get value
        for j in range(0, num_samples):
            for i, m in enumerate(self.models):
                samp = m._sample_param_posterior(num_samples=1)
                envs.append(m._set_model_parameters_from_sample(
                    t, copy.deepcopy(m.odesys), 0, samp))
            enviro = Multiplume(envs)
            vals.append(enviro.get_value(t, loc))
        return np.nanmean(vals, axis=0), np.nanvar(vals, axis=0)

    def get_snapshot(self, t, z=None, xrange=None, yrange=None, xres=None,
                     yres=None, from_cache=False):
        """Get a deterministic prediction of full state at time t.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            from_cache (bool): if True, return nearest value from cache

        Returns: P
        """
        # only want probability of being in plume
        P = self.enviro.get_snapshot(t, z, xrange, yrange, xres, yres,
                                     from_cache=from_cache)
        return P

    def get_maxima(self, t, z=None, xrange=None, yrange=None, xres=None,
                   yres=None, from_cache=False):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time
            z (list[float]): heights
            xrange (tuple[float]): xdim range
            yrange (tuple[float]): ydim range
            xres (int): xdim resolution
            yres (int): ydim resolutions
            from_cache (bool): whether to read from cache

        Returns: list[float tuple] maximum location
        """
        # Get snapshot
        loc = self.enviro.get_maxima(t, z, xrange, yrange, xres, yres,
                                     from_cache=from_cache)
        return loc

    def get_snapshot_prediction(self, t, z=None, xrange=None, yrange=None,
                                xres=None, yres=None, from_cache=False,
                                num_samples=500):
        """Get model output t and loc.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            from_cache (bool): if True, return nearest value from cache
            num_samples (int): number of posterior samples to use

        Returns: means at heights, variances at heights
        """
        if from_cache:
            # Read from cache and return nearest values in xyzt
            ms, vs, xs, ys, zs, ts, ns = self.read_prediction_cache()
            mean = np.zeros((len(z), xres, yres))
            var = np.zeros((len(z), xres, yres))

            # Get nearest t in snapshots
            tq = np.fabs(ts - t)
            idt = np.argmin(tq)
            for i, zi in enumerate(z):
                # Get nearest z in snapshots
                zq = np.fabs(zs - zi)
                idz = np.argmin(zq)

                # Now have the mean and variance
                mean[i, :, :] = ms[idt, :, :, idz]
                var[i, :, :] = vs[idt, :, :, idz]
            return mean, var

        # If not using cache, compute and return
        mean_heights = np.zeros((len(z), xres, yres))
        var_heights = np.zeros((len(z), xres, yres))
        # Push samples through get value
        for j in range(0, num_samples):
            envs = []
            for i, m in enumerate(self.models):
                samp = m._sample_param_posterior(num_samples=1)
                envs.append(m._set_model_parameters_from_sample(
                    t, copy.deepcopy(m.odesys), 0, samp))
            enviro = Multiplume(envs)
            for k, zh in enumerate(z):
                P = enviro.get_snapshot(t,
                                        [zh],
                                        xrange,
                                        yrange,
                                        xres,
                                        yres,
                                        from_cache=from_cache)
                # compute mean and variance in rolling fashion
                mean_old = copy.deepcopy(mean_heights[k, :, :])
                mean_heights[k, :, :] = mean_old + (P - mean_old) / (i + 1)
                var_heights[k, :, :] = var_heights[k, :, :] + \
                    (P - mean_old) * (P - mean_heights[k, :, :]) / (i + 1)
        return mean_heights, var_heights
