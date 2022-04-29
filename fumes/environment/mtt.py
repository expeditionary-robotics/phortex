"""Interface for science model-driven environments."""

from math import isinf
import os
import copy
import warnings
import itertools
from logging import raiseExceptions
from shutil import copyfile
import numpy as np
from numpy.matlib import repmat
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import interpolate
from scipy.stats import multivariate_normal
from scipy.interpolate import RegularGridInterpolator

from shapely.geometry import LineString, Point
import shapely.speedups
if shapely.speedups.available:
    shapely.speedups.enable()
from netCDF4 import Dataset

import gpytorch as gpy
import torch

from fumes.utils import data_home, output_home, tic, toc
from fumes.environment.environment import Environment
from fumes.environment.utils import bent_speer_rona, speer_rona, \
    tohidi_kaye, rotgauss, pacific_sp_S, pacific_sp_T, eos_rho, \
    ncdump, fileno, stdout_redirected, random_str
from fumes.environment.utils import ExactGPModel, StandardApproximateGP

from fumes.simulator.utils import draw_ellipse, to_homogenous, translation, \
    xrotation, zrotation


class StationaryMTT(Environment):
    """A hydrothermal plume model, without crossflow.

    Developed from equations in:
    - K. G. Speer and P. A. Rona. "A model of
      an Atlantic and Pacific hydrothermal plume," Journal of
      Geophysical Research: Oceans, vol.94, no. C5, pp.6213-6220, 1989.
    - B. Morton, G. I. Taylor, and J. S. Turner, "Turbulent
      graviational convection from maintained and instantaneous sources,"
      Proceedings of the Royal Society of London. Series A. Mathematical
      and Physical Sciences, vol. 234, no. 1196, pp. 1-23, 1956.
    """

    def __init__(self, extent, plume_loc, z, tprof, sprof, rhoprof, vex=0.1, area=0.1,
                 density=1000, salt=34.608, temp=300, entrainment=0.255):
        """Initializes an MTT model class from Speer and Rona."""
        self.NAME = f"StationaryMTT{random_str()}"

        # Physical constants
        self.extent = extent
        self.entrainment = entrainment  # entrainment coefficient
        self.g = -9.81  # acceleration due to gravity N/kg
        self.loc = plume_loc  # (easting, northing, height) coordinates

        # Inputs of source
        self.z = z + plume_loc[-1]  # global heights to integrate over in m
        self.v0 = vex  # initial plume exit velocity
        self.a0 = area  # area of plume orifice
        self.rho0 = density  # intial density of the plume in kg/m^3
        self.s0 = salt  # absolute salinity at source in % (unitless)
        self.t0 = temp  # potential temperature at source in C

        self.tprof = tprof  # background temperature profile function
        self.sprof = sprof  # background salinity profile function
        self.rhoprof = rhoprof  # background density profile function

        # Initialize internal variables
        self.cache_name = None
        self.gp_cache_name = None
        self._gp_model = None
        self._model = {}

        # Create solution
        self.solve(t=0.0)

    def set_name(self, name):
        """Overwrites the default model name with input."""
        self.NAME = name

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

    def get_parameters(self):
        """Returns the parameters defining the model."""
        print("Exit Velocity: ", self.v0)
        print("Orifice Area: ", self.a0)
        print("Source Temp (C) and Salinity: ", self.t0, self.s0)
        print("Source Density: ", self.rho0)
        print("Entrainment coeff: ", self.entrainment)
        print("Gravity :", self.g)
        return (self.v0, self.a0, self.t0, self.s0, self.rho0,
                self.entrainment, self.g)

    def _json_stats(self):
        """Creates a dict object about model information for JSON.

        Returns:
            dict object compatible for JSON saving
        """
        json_config_dict = {"model_type": "stationary",
                            "plume_loc": self.loc,
                            "extent": self.extent.get_attributes(),
                            "temp": self.t0,
                            "salt": self.s0,
                            "area": self.a0,
                            "velocity": self.v0,
                            "density": self.rho0,
                            "entrainment": self.entrainment,
                            "z": self.z.tolist(),
                            "tprof": self.tprof(self.z).tolist(),
                            "sprof": self.sprof(self.z).tolist(),
                            "rhoprof": self.rhoprof(self.tprof(self.z), self.sprof(self.z)).tolist(),
                            }
        return json_config_dict

    def get_maxima(self, t, z=None, xrange=None, yrange=None, xres=None,
                   yres=None, from_cache=False):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time

        Returns: tuple[float] maximum location
        """
        xrange, xres, yrange, yres, z = \
            self.set_extent_with_z(xrange, xres, yrange, yres, z)

        # Get snapshot
        val, x, y, z = self.get_snapshot_with_grid(
            t, z, xrange, yrange, xres, yres, return_all=False,
            from_cache=from_cache)
        # plt.imshow(val[0,:,:], origin="lower")
        # plt.show()

        # Return coordinates of maxima
        zi, xi, yi = np.unravel_index(val.argmax(), val.shape)
        return (x[xi, yi], y[xi, yi], z[zi])

    def solve(self, t, overwrite=False):
        """Given class parameter settings, compute the spatial solution.

        Args:
            t (float): the global time
            overwrite (bool): if True, will overwrite existing cache.
        """
        if t in self._model and not overwrite:
            # If time t has already  been solved, return
            return

        params = (self.rho0, self.sprof, self.tprof, self.rhoprof,
                  self.entrainment, self.g)
        init_cond = [self.v0, self.a0, self.s0, self.t0]
        with stdout_redirected():
            sol = odeint(speer_rona, init_cond, self.z, args=params)

        # The solution may contain post-neutral buoyancy "junk", remove
        idx = np.where(sol[:, 0] <= 0.001)
        if idx[0].shape[0] != 0:
            warnings.warn("Warning: Runaway integration, setting values NaN.")
            sol[idx[0][0]:, :] = np.nan

        # Add to model at time t
        self._model[t] = sol

    def model(self, t):
        """Get model solution at time t."""
        # Stationary plume, so just use solution for time t=0
        t = 0.0
        assert(t in self._model)
        return self._model[t]

    def get(self, t, variable):
        """Getter for model attributes at time t."""
        func = getattr(self, variable)
        return func(t)

    def vel(self, t):
        """Getter for model velocity at time t."""
        return self.model(t)[:, 0]

    def area(self, t):
        """Getter for model area at time t."""
        return self.model(t)[:, 1]

    def a(self, t):
        """Getter for model radius at time t."""
        return np.sqrt(self.area(t) / np.pi)

    def sal(self, t):
        """Getter for model salinity at time t."""
        return self.model(t)[:, 2]

    def temp(self, t):
        """Getter for model temperature at time t."""
        return self.model(t)[:, 3]

    def envelope(self, t):
        """Returns the left extent, centerline, and right extent at time t."""
        le = -self.a(t)
        re = self.a(t)
        cl = np.zeros_like(le)
        return le, cl, re

    def write_cache(self, tvec, xrange=None, yrange=None, zrange=None,
                    xres=None, yres=None, zres=None, overwrite=True):
        """Generates xyzt snapshots to cache in memory for reference.

        Args:
            tvec (array[float]): time vector
            xrange (tuple[float]): min/max xdim to simulate
            yrange (tuple[float]): min/max ydim to simulate
            zrange (tuple[float]): min/max zdim to simulate
            xres (int): xdim resolution
            yres (int): ydim resolution
            zres (int): zdim resolution
            overwrite (bool): if True, will overwrite existing cache.
        """
        joint_locs = None
        joint_vals = None

        xrange, xres, yrange, yres, zrange, zres = \
            self.set_extent(xrange, xres, yrange, yres, zrange, zres)

        # Setup file to write with standard name
        self.cache_name = f"{self.NAME}_t{tvec[0]}-{tvec[-1]}_" \
            f"x{xrange[0]}-{xrange[1]}_" \
            f"xres{xres}_" \
            f"y{yrange[0]}-{yrange[1]}_" \
            f"yres{yres}_" \
            f"z{zrange[0]}-{zrange[1]}_" \
            f"zres{zres}.nc"
        self.cache_name = os.path.join(output_home(), "simulations", self.cache_name)

        # Check if cache already exists
        if os.path.exists(self.cache_name) and overwrite:
            warnings.warn("Cache already exists. Overwriting ...")
            os.remove(self.cache_name)
        if os.path.exists(self.cache_name) and not overwrite:
            warnings.warn("Cache already exists. Returning without writing.")
            return

        ncfile = Dataset(self.cache_name, "w", format="NETCDF4")
        ncfile.createDimension("time", None)
        ncfile.createDimension("x", xres)
        ncfile.createDimension("y", yres)
        ncfile.createDimension("z", zres)

        # Create variables to save data to
        times = ncfile.createVariable("time", "f8", ("time",))
        xs = ncfile.createVariable("x", "f8", ("x",))
        ys = ncfile.createVariable("y", "f8", ("y",))
        zs = ncfile.createVariable("z", "f8", ("z",))
        vals = ncfile.createVariable("val", "f8", ("time", "x", "y", "z",))

        # Create variable values
        x = np.linspace(xrange[0], xrange[1], xres)
        y = np.linspace(yrange[0], yrange[1], yres)
        z = np.linspace(zrange[0], zrange[1], zres)
        xg, yg, zg = np.meshgrid(x, y, z, indexing="ij")

        # Populate array indices
        times[:] = np.asarray(tvec)  # save time coords
        xs[:] = x  # save x coords
        ys[:] = y  # save y coords
        zs[:] = z  # save z coords

        # Populate data
        for i, t in enumerate(tvec):
            print(f"Time {t}, {i} of {len(tvec)}.")
            P = self.get_snapshot(
                t=t,
                z=z,
                xrange=xrange,
                yrange=yrange,
                xres=xres,
                yres=yres,
                return_all=False,
                from_cache=False)

            for j, zm in enumerate(np.linspace(zrange[0], zrange[1], zres)):
                vals[i, :, :, j] = P[j, :, :].reshape(xres, yres)

            locs = np.hstack([t * np.ones(xg.shape).reshape(-1, 1),
                              xg.reshape(-1, 1),
                              yg.reshape(-1, 1),
                              zg.reshape(-1, 1)])
            # Append current location and values to the queue
            if joint_locs is None:
                joint_locs = copy.copy(locs)
            else:
                joint_locs = np.vstack([joint_locs, locs])
            if joint_vals is None:
                joint_vals = copy.copy(vals[i, :, :, :].reshape(-1,))
            else:
                joint_vals = np.vstack([joint_vals, vals[i, :, :, :].reshape(-1,)])

        # Close the ncfile
        ncfile.close()

        # Return the joint set of points and values cached
        return joint_locs, joint_vals

    def write_gp_cache(self, tvec, xrange=None, yrange=None,
                       zrange=None, xres=None, yres=None,
                       zres=None, overwrite=True,
                       seed_kernel=None, visualize=False):
        """Generates GP to cache in memory for reference.

        Args:
            tvec (array[float]): time vector
            xrange (tuple[float]): min/max xdim to simulate
            yrange (tuple[float]): min/max ydim to simulate
            zrange (tuple[float]): min/max zdim to simulate
            xres (int): xdim resolution
            yres (int): ydim resolution
            zres (int): zdim resolution
            overwrite (bool): if True, will overwrite existing cache.
            seed_kernel (str): if not None, will seed the kernel of the
                written GP with the contents of the seed_kernel filename
        """
        # Get xranges
        xrange, xres, yrange, yres, zrange, zres = \
            self.set_extent(xrange, xres, yrange, yres, zrange, zres)

        # Write location and data to the cache
        locs, vals = self.write_cache(
            tvec, xrange, yrange, zrange,
            xres, yres, zres, overwrite=overwrite)

        # Convert to torch tensors for GP training
        locs = torch.tensor(locs).float()
        vals = torch.tensor(vals).float()

        # Setup file to write with standard name
        self.gp_cache_name = f"GP-{self.NAME}_" \
            f"t{tvec[0]}-{tvec[-1]}_" \
            f"x{xrange[0]}-{xrange[1]}_" \
            f"xres{xres}_" \
            f"y{yrange[0]}-{yrange[1]}_" \
            f"yres{yres}_" \
            f"z{zrange[0]}-{zrange[1]}_" \
            f"zres{zres}.pth"

        self.gp_cache_name = os.path.join(
            output_home(), "simulations", self.gp_cache_name)
        if seed_kernel is not None:
            kernel_cache_name = os.path.join(
                output_home(), "simulations", seed_kernel)

        # Check if cache already exists
        if os.path.exists(self.gp_cache_name) and overwrite:
            warnings.warn("Cache already exists. Overwriting ...")
            os.remove(self.gp_cache_name)
        if os.path.exists(self.gp_cache_name) and not overwrite:
            warnings.warn("Cache already exists. Returning without writing.")
            return
        if seed_kernel is not None and os.path.exists(kernel_cache_name) and not overwrite:
            warnings.warn("Cache already exists. Returning without writing.")
            # Copy the seed kernel file to the current GP file
            copyfile(kernel_cache_name, self.gp_cache_name)
            return

        # Populate data
        print("Initializing GP model. ")
        gp_model = self._train_gp(locs, vals.reshape(-1,), train=True)

        if visualize:
            # Validate performance of the trained model.
            with torch.no_grad(), gpy.settings.fast_pred_var():
                l = locs.reshape(len(tvec), xres, yres, zres, 4)  # txyz
                v = vals.reshape(len(tvec), xres, yres, zres)

                # Get the GP mean forecast
                v_hat = gp_model(locs).mean
                v_hat = v_hat.reshape(len(tvec), xres, yres, zres)

                # Select a specific slice for plotting
                v_hat = v_hat[0, :, :, 3]
                v = v[0, :, :, 3]
                lx = l[0, :, :, 3, 1]
                ly = l[0, :, :, 3, 2]

                # Plot value
                fig, axs = plt.subplots(1, 2)
                axs[0].scatter(lx, ly, v, levels=1000)
                axs[0].set_xlim(xrange[0], xrange[1])
                axs[0].set_ylim(yrange[0], yrange[1])

                axs[1].scatter(lx, ly, v_hat, levels=1000)
                axs[1].set_xlim(xrange[0], xrange[1])
                axs[1].set_ylim(yrange[0], yrange[1])
                plt.show()

    def _train_gp(self, locs, vals, train=False):
        """ Train the kernel parameters using a Gaussian process model
        and the marginal log likelihood.

        Args:
            locs (torch.tensor): the input states
            vals (torch.tenor): rewards of the input states
            train (bool): if False, attempts to load model parameters
                from file
        """
        likelihood = gpy.likelihoods.GaussianLikelihood()

        # Write relevent variables to cuda
        if torch.cuda.is_available():
            likelihood = likelihood.cuda()
            locs = locs.cuda()
            vals = vals.cuda()

        # Initialize the GP model in order to train the kernel
        model = ExactGPModel(locs, vals, likelihood, num_dims=4,
                             name=self.gp_cache_name)
        # model = StandardApproximateGP(inducing_points=locs[0:-1:50, :],
        #                               name=self.gp_cache_name)

        if torch.cuda.is_available():
            model = model.cuda()

        if not train and os.path.isfile(model.model_file):
            # Load from file
            model_state = torch.load(model.model_file)
            model.load_state_dict(model_state)
        else:
            # # Initial values for kernel parameters
            hypers1 = {
                'covar_module.base_kernel.lengthscale': torch.tensor(10),
                'covar_module.outputscale': torch.tensor(2.0),
                'likelihood.noise_covar.noise': torch.tensor(0.01),
            }
            # hypers2 = {
            #     'noise_covar.noise': torch.tensor(0.01),
            # }

            # Initialize model
            model.initialize(**hypers1)
            # likelihood.initialize(**hypers2)

            print("Start training.")

            # Find optimal model hyperparameters
            # Put model and likelihood in train mode
            model.train()
            # likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam([
                # Includes GaussianLikelihood parameters
                {'params': model.parameters()},
                # {'params': likelihood.parameters()},
                # {'params': model.covar_module.base_kernel.lengthscale},
            ], lr=0.1)

            # "Loss" for GPs - the marginal log likelihood
            if type(model).__name__ == "ExactGPModel":
                mll = gpy.mlls.ExactMarginalLogLikelihood(likelihood, model)
            elif type(model).__name__ == "StandardApproximateGP":
                mll = gpy.mlls.VariationalELBO(
                    likelihood, model, num_data=locs.shape[0])
            else:
                raise ValueError(
                    f"Unrecognized model type {type(model).__name__}.")

            training_iter = 50
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()

                # Output from model
                output = model(locs)

                # Calc loss and backprop gradients
                loss = -mll(output, vals)

                loss.backward()
                print("Iter %d/%d - Loss: %.3f \r\n"
                      "\tlengthscale1: %.3f \r\n"
                      "\tlengthscale2: %.3f \r\n"
                      "\tlengthscale3: %.3f \r\n"
                      "\tlengthscale4: %.3f \r\n"
                      "noise: %.3f   var: %.3f" % (
                          i + 1, training_iter, loss.item(),
                          model.covar_module.base_kernel.lengthscale[0][0].item(),
                          model.covar_module.base_kernel.lengthscale[0][1].item(),
                          model.covar_module.base_kernel.lengthscale[0][2].item(),
                          model.covar_module.base_kernel.lengthscale[0][3].item(),
                          likelihood.noise.item(), model.covar_module.outputscale.item()
                      ))
                optimizer.step()

            print("Trained parameters:")
            torch.save(model.state_dict(), model.model_file)

        print("Kernel parameters:")
        print("\tlengthscale1: %.3f \r\n"
              "\tlengthscale2: %.3f \r\n"
              "\tlengthscale3: %.3f \r\n"
              "\tlengthscale4: %.3f \r\n"
              "noise: %.3f   var: %.3f" % (
                  model.covar_module.base_kernel.lengthscale[0][0].item(),
                  model.covar_module.base_kernel.lengthscale[0][1].item(),
                  model.covar_module.base_kernel.lengthscale[0][2].item(),
                  model.covar_module.base_kernel.lengthscale[0][3].item(),
                  likelihood.noise.item(), model.covar_module.outputscale.item()
              ))
        # likelihood.eval()
        model.eval()
        return model

    def read_cache(self, verbose=False):
        """Given a model cache, read the data.

        Args
            verbose (bool): whether to print ncinfo

        Returns
            pc, xc, yc, zc, tc values from model snapshots
        """
        # Check that the current cache is populated
        assert(self.cache_name is not None)
        ncfile = Dataset(self.cache_name, "r")
        if verbose:
            ncdump(ncfile)

        pc = ncfile.variables['val'][:]  # data
        xc = ncfile.variables['x'][:]  # xcoords
        yc = ncfile.variables['y'][:]  # ycoords
        zc = ncfile.variables['z'][:]  # zcoords
        tc = ncfile.variables['time'][:]  # tcoords

        # Close the ncfile
        ncfile.close()

        return pc, xc, yc, zc, tc

    def read_gp_cache(self, verbose=False, overwrite=True):
        """Given a model cache, read the data.

        Args
            verbose (bool): whether to print ncinfo

        Returns
            pc, xc, yc, zc, tc values from model snapshots
        """
        # Check that the current cache is populated
        assert(self.gp_cache_name is not None)
        p, x, y, z, t = self.read_cache(verbose)

        # Create variable values
        x = torch.tensor(x).float()
        y = torch.tensor(y).float()
        z = torch.tensor(z).float()
        t = torch.tensor(t).float()
        vals = torch.tensor(p).float().reshape(-1,)
        tg, xg, yg, zg = torch.meshgrid(t, x, y, z, indexing="ij")

        locs = torch.hstack([tg.reshape(-1, 1),
                             xg.reshape(-1, 1),
                             yg.reshape(-1, 1),
                             zg.reshape(-1, 1)])

        # Load the pretrained model by setting train to False
        self.gp_model = self._train_gp(locs, vals, train=False)
        return self.gp_model

    def get_value(self, t, loc, return_all=False, from_cache=False,
                  cache_interp="gp"):
        """Get a deterministic prediction of the model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]) or list(tuple[float]): a location or
                list of locations, in xyz
            return_all (bool): whether to return more than prob
            from_cache (bool): if True, return nearest value from cache
            cache_interp (str): Either "gp" or "lookup", for Gaussian
                process or look-up table interpolation.

        Returns: tuple(float) plume probability and/or V, S, T
            (probability must be returned as the first element of the tuple)
        """
        if from_cache and return_all:
            warnings.warn(
                "Currently do not support caching of V, S, T."
                "Only probability will be returned.")

        if from_cache:
            # Read from cache and return nearest values in xyzt
            qx, qy, qz = loc
            qx = np.atleast_1d(qx)
            qy = np.atleast_1d(qy)
            qz = np.atleast_1d(qz)
            prob = np.zeros_like(qx)

            if cache_interp == "lookup":
                print("Lookup time.")
                ps, xs, ys, zs, ts = self.read_cache()

                # Get nearest t in snapshots
                tq = np.fabs(ts - t)
                idt = np.argmin(tq)
                zq = np.fabs(zs - qz)
                idz = np.argmin(zq)

                for i, (lx, ly) in enumerate(zip(qx, qy)):
                    # Get nearest x in snapshots
                    xq = np.fabs(xs - lx)
                    idx = np.argmin(xq)
                    # Get nearest y in snapshots
                    yq = np.fabs(ys - ly)
                    idy = np.argmin(yq)
                    # Now have the probability
                    prob[i] = ps[idt, idx, idy, idz]
                return prob
            elif cache_interp == "gp":
                if self._gp_model is None:
                    # If we havn't set the model from the cache
                    # yet, do so
                    print("Reading from gp cache.")
                    self._gp_model = self.read_gp_cache()

                # Fast predictions
                with torch.no_grad(), gpy.settings.fast_pred_var():
                    qloc = torch.hstack([
                        t * torch.ones(qx.shape).reshape(-1, 1),
                        torch.tensor(qx).reshape(-1, 1),
                        torch.tensor(qy).reshape(-1, 1),
                        torch.tensor(qz).reshape(-1, 1)]).float()
                    prob = self._gp_model(qloc).mean
                return prob
            else:
                raise ValueError(
                    f"Unsupported interpolaition: {cache_interp}")

        # If not read from cache, solve and return value
        self.solve(t, overwrite=True)
        prob = self._interpolate_loc(t, loc, return_all)
        return prob

    def _interpolate_loc(self, t, loc, return_all):
        """Return model values at location by interpolating model solution.

        Args:
            t (float): the global time
            loc (tuple[float]) or list(tuple[float]): a location or
                list of locations, in xyz
            return_all (bool): whether to return more than prob
        """
        qx, qy, z = loc
        qx = np.atleast_1d(qx)
        qy = np.atleast_1d(qy)
        z = np.atleast_1d(z)

        # Convert to global frame
        x = qx - self.loc[0]
        y = qy - self.loc[1]
        # Note that z and self.z are both in global coordinates

        # Get plume information at height
        if return_all:
            Vv = np.zeros_like(qx)
            Sv = np.zeros_like(qx)
            Tv = np.zeros_like(qx)
        probv = np.zeros_like(qx)

        plume = interpolate.interp1d(self.z, self.model(t), axis=0)

        for i, (xx, yy, zz) in enumerate(zip(x, y, z)):
            if zz < self.loc[-1]:
                # the value is below the plume source, set everything to 0
                # or background
                if return_all:
                    Vv[i] = 0
                    Sv[i] = self.sprof(zz)
                    Tv[i] = self.tprof(zz)
                probv[i] = 0
            else:
                V, A, S, T = plume(zz)

                rad = np.sqrt(A / np.pi)

                # TODO: consider other settings here
                if np.isnan(rad):
                    warnings.warn("Warning: Radius is nan, setting to 1000m.")
                    rad = 1000.

                # Assume a gaussian in xy
                prob = multivariate_normal(
                    mean=[0, 0], cov=[[rad**2, 0], [0, rad**2]])

                spread = np.exp(-(xx**2 / (2 * rad**2) + yy**2 /
                                (2 * rad**2)))

                if return_all:
                    Vv[i] = V * spread
                    Sv[i] = S * spread
                    Tv[i] = T * spread
                probv[i] = prob.pdf(np.asarray([xx, yy]).T)

        if return_all:
            return probv, Vv, Sv, Tv
        return probv

    def get_snapshot(self, t, z=None, xrange=None, yrange=None, xres=None,
                     yres=None, return_all=False, from_cache=False):
        """Get a deterministic prediction of full state at time t.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            return_all (bool): whether to return prob or all values
            from_cache (bool): if True, return nearest value from cache

        Returns: z x nx x ny array of P and/or V, S, T
            (probability must be returned as the first element of the tuple)
        """
        vals, x, y, z = self.get_snapshot_with_grid(
            t, z, xrange, yrange, xres, yres, return_all, from_cache)
        return vals

    def get_snapshot_with_grid(self, t, z=None, xrange=None, yrange=None, xres=None,
                               yres=None, return_all=False, from_cache=False):
        """Get a deterministic prediction of full state at time t and
        snapshot grid.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            return_all (bool): whether to return prob or all values
            from_cache (bool): if True, return nearest value from cache

        Returns: z x nx x ny array of P and/or V, S, T
            (probability must be returned as the first element of the tuple)
        """
        # Initialize variables
        if z is None and "StationaryMTT" in self.NAME:
            z = self.z
        elif z is None and "CrossflowMTT" in self.NAME:
            z = self.z_disp(t)

        xrange, xres, yrange, yres, z = \
            self.set_extent_with_z(xrange, xres, yrange, yres, z)
        print(z)
        P = np.zeros((len(z), xres, yres))

        if return_all:
            var_setup = False
            var_list = []

        if from_cache and return_all:
            warnings.warn("Currently do not support caching of V, S, T."
                          "Only probability will be returned.")

        if from_cache:
            # TODO: only lookup is supported for get_snapshot
            # should add GP cache compatability
            ps, xs, ys, zs, ts = self.read_cache()

            # Get nearest t in snapshots
            tq = np.fabs(ts - t)
            idt = np.argmin(tq)
            for i, zi in enumerate(z):
                # Get nearest z in snapshots
                zq = np.fabs(zs - zi)
                idz = np.argmin(zq)

                # Now have the probability
                P[i, :, :] = ps[idt, :, :, idz]
            return (P, xs, ys, zs)
        else:
            nx, ny = np.meshgrid(
                np.linspace(xrange[0], xrange[1], xres),
                np.linspace(yrange[0], yrange[1], yres))

            for i, h in enumerate(z):
                res = self.get_value(
                    t,
                    (nx.flatten(),
                     ny.flatten(),
                     np.ones_like(nx.flatten()) * h),
                    return_all=return_all,
                    from_cache=from_cache)

                # Extract probability from result
                if type(res) is tuple:
                    pt = res[0]
                else:
                    pt = res

                P[i, :, :] = pt.reshape(xres, yres)

                # Setup empty return values
                if return_all and not var_setup:
                    # Iterate through all returns except probability
                    for var in res[1:]:
                        var_list.append(np.zeros((len(z), xres, yres)))

                if return_all:
                    for j, var in enumerate(res[1:]):
                        var_list[j][i, :, :] = var.reshape(xres, yres)

        if return_all:
            # Return probability and unpacked variable list
            return ((P, *var_list), nx, ny, z)
        return (P, nx, ny, z)

    def get_pointcloud(self, t):
        """Return a point cloud of 3D plume structure at time t."""
        self.solve(t=t, overwrite=False)
        area1 = area2 = self.a(t=t)
        x_loc = np.zeros_like(area1)
        z_loc = self.z

        pts_all = None
        for i, (a, b, x, z) in enumerate(
                zip(area1, area2, x_loc, z_loc)):
            pts = draw_ellipse(a, b)
            pts = to_homogenous(pts[0], pts[1])
            T = translation(self.loc[0], self.loc[1], z)

            # Apply affine transformation to points
            pts_p = pts @ T
            if pts_all is None:
                pts_all = pts_p
            else:
                pts_all = np.vstack([pts_all, pts_p])

        return pts_all


class CrossflowMTT(StationaryMTT):
    """Creates an along-plume rise model with crossflow.

    Developed from equations in:
    - A. Tohidi and N. B. Kaye, "Highly buoyant bent-over plumes
      in a boundary layer," Atmospheric Environment, vol. 131,
      pp. 97-114, 2016.
    - G. Xu and D. Di Iorio, "Deep sea hydrothermal plumes and their
      interactions with oscillatory flows," Geochemistry, Geophysics,
      Geosystems, vol. 13, no. 9, 2012.
    """

    def __init__(self, extent, plume_loc, s, curfunc, headfunc, tprof, sprof, rhoprof,
                 vex=0.1, area=0.1, density=1000, salt=34.608, temp=300,
                 lam=1.0, entrainment=[0.011, 0]):
        """Initializes a Tohidi-Kaye model class."""
        self.NAME = f"CrossflowMTT{random_str()}"

        # Physical constants
        self.extent = extent
        self.entrainment = entrainment  # entrainment coefficient
        self.g = -9.81  # acceleration due to gravity N/kg
        self.loc = plume_loc  # (easting, northing, height) source coordinates

        # Inputs of source
        self.s = s  # along-axis distance to integrate over in m
        self.z_offset = plume_loc[-1]  # the absolute altitude in the world
        self.v0 = vex  # initial plume exit velocity
        self.a0 = area  # area of plume orifice
        self.tprof = tprof  # background temperature profile function
        self.sprof = sprof  # background salinity profile function
        self.rhoprof = rhoprof  # background density profile function
        self.rho0 = density  # intial density of the plume in kg/m^3
        self.s0 = salt  # absolute salinity at source in % (unitless)
        self.t0 = temp  # potential temperature at source in C
        self.currents = curfunc  # function that describes currents
        self.heading = headfunc  # function that describes current heading

        self.lam = lam  # ratio of source major-minor axis
        self.q0 = self.lam * self.a0 / np.pi * self.v0  # source heat flux
        self.m0 = self.q0 * self.v0  # source momentum flux
        # source buoyancy flux
        self.f0 = -self.g * 10**(-4) * (self.t0 - self.tprof(0)) * self.q0
        self.th0 = np.pi / 2.  # source output angle

        # Initialize internal variables
        self.cache_name = None
        self._model = {}
        self._gp_model = None

        # Create solution
        self.solve(t=0.0)

    def _json_stats(self):
        """Creates a dict object about model information for JSON.

        Returns:
            dict object compatible for JSON saving
        """
        t = np.linspace(0, 3600 * 24, 25)
        z = np.linspace(0, 200, 100)
        json_config_dict = {"model_type": "crossflow",
                            "plume_loc": self.loc,
                            "extent": self.extent.get_attributes(),
                            "temp": self.t0,
                            "salt": self.s0,
                            "area": self.a0,
                            "vex": self.v0,
                            "density": self.rho0,
                            "entrainment": self.entrainment,
                            "lam": self.lam,
                            "z": z.tolist(),
                            "s": self.s.tolist(),
                            "t": t.tolist(),
                            "tprof": self.tprof(self.z).tolist(),
                            "sprof": self.sprof(self.z).tolist(),
                            "rhoprof": self.rhoprof(self.tprof(self.z), self.sprof(self.z)).tolist(),
                            "curfunc": self.currents(None, t).tolist(),
                            "headfunc": self.heading(t).tolist(),
                            }
        return json_config_dict

    def solve(self, t, overwrite=False):
        """Given class parameter settings, compute the spatial solution.

        Args:
            t (float): the global time
            overwrite (bool): if True, will overwrite existing cache.
        """
        if t in self._model and not overwrite:
            # If time t has already  been solved, return
            return

        # Update params dependent on others
        self.q0 = self.lam * self.a0 / np.pi * self.v0
        self.m0 = self.q0 * self.v0
        self.f0 = -self.g * 10**(-4) * (self.t0 - self.tprof(0)) * self.q0
        # solve for the stationary plume
        params = (t, self.rho0, self.currents, self.tprof, self.sprof,
                  self.rhoprof, self.lam, self.entrainment,
                  self.g, self.z_offset)
        # conditions are in order: Q, M, F, theta, x, z
        init_cond = [self.q0, self.m0, self.f0, self.th0, 0., 0.]
        sol = odeint(tohidi_kaye, init_cond, self.s, args=params)

        # Add to model at time t
        self._model[t] = sol

    def model(self, t):
        """Get model solution at time t."""
        if t in self._model:
            return self._model[t]
        else:
            self.solve(t=t)
            return self._model[t]

    def _interpolate_loc(self, t, loc, return_all):
        """Return model values at location by interpolating model solution.

        Args:
            t (float): the global time
            loc (tuple[float]) or list(tuple[float]): a location or
                list of locations, in xyz
            return_all (bool): whether to return more than prob
        """
        prob = []

        QueryX, QueryY, z = loc

        # Re-center query to plume
        Qx = np.atleast_1d(QueryX - self.loc[0])
        Qy = np.atleast_1d(QueryY - self.loc[1])
        Qz = np.atleast_1d(z)

        X = self.x_disp(t)
        Z = self.z_disp(t)  # put plume in world coords
        line = LineString(zip(X, Z))
        plume = interpolate.interp1d(X, self.model(t), axis=0)
        for x, y, z in zip(Qx, Qy, Qz):
            # convert to plume coordinates
            qx = np.cos(self.heading(t)) * x + np.sin(self.heading(t)) * y
            qy = -np.sin(self.heading(t)) * x + np.cos(self.heading(t)) * y
            qz = z
            # find the closest point in the plume
            p = Point(qx, qz)
            closest_point = line.interpolate(line.project(p))
            # interpolate model to get the right a, b
            try:
                Q, M, F, Th, Xx, Zx = plume(closest_point.x)
            except IndexError:
                Q, M, F, Th, Xx, Zx = plume(X[-1])
            Zx = Zx + self.z_offset  # put z in absolute coordinates
            a = Q / np.sqrt(self.lam * M)
            b = self.lam * a
            if np.isnan(a) or np.isinf(a):
                a = 1000
                b = self.lam * 1000
            # put query into ellipse frame
            eqx = qx - Xx
            eqy = qy
            eqz = np.sqrt(eqx**2 + (qz - Zx)**2)
            try:
                p = multivariate_normal(
                    mean=[0, 0], cov=[[a**2, 0], [0, b**2]])
                prob.append(p.pdf(np.asarray([eqz, eqy]).T))
            except Exception as e:
                # import pdb; pdb.set_trace()
                prob.append(0.0)

        return np.asarray(prob)

    def q(self, t):
        """Getter for model volume transport at time t."""
        return self.model(t)[:, 0]

    def m(self, t):
        """Getter for model momentum flux at time t."""
        return self.model(t)[:, 1]

    def f(self, t):
        """Getter for buoyancy flux at time t."""
        return self.model(t)[:, 2]

    def theta(self, t):
        """Getter for plume angle at time t."""
        return self.model(t)[:, 3]

    def x_disp(self, t):
        """Getter for plume x displacement (current aligned) at time t."""
        return self.model(t)[:, 4]

    def z_disp(self, t):
        """Getter for plume z rise (from source) at time t."""
        return self.model(t)[:, 5] + self.z_offset

    def area(self, t):
        """Getter for orthogonal plume area at location s at time t."""
        return np.pi * self.a(t) * self.b(t)

    def a(self, t):
        """Getter for orthagonal plume axis A at time t."""
        return self.q(t) / np.sqrt(self.lam * self.m(t))

    def b(self, t):
        """Getter for orthogonal plume axis B at time t."""
        return self.lam * self.a(t)

    def envelope(self, t):
        """Returns the left extent, centerline, and right extent at time t."""
        return (self.x_disp(t) - self.a(t) * np.sin(self.theta(t)),
                self.z_disp(t) + self.a(t) * np.cos(self.theta(t))), \
            (self.x_disp(t), self.z_disp(t)), \
            (self.x_disp(t) + self.a(t) * np.sin(self.theta(t)),
             self.z_disp(t) - self.a(t) * np.cos(self.theta(t)))

    def vel(self, t):
        """Getter for model velocity at time t."""
        raise NotImplementedError

    def sal(self, t):
        """Getter for model salinity at time t."""
        raise NotImplementedError

    def temp(self, t):
        """Getter for model temperature at time t."""
        raise NotImplementedError

    def local_to_global(self, t, x, z):
        """Convert local points in x, z format to the global reference frame."""
        x_p = x * np.cos(self.heading(t)) + self.loc[0]
        y_p = x * np.sin(self.heading(t)) + self.loc[1]
        z_p = copy.copy(z)
        return x_p, y_p, z_p

    def get_pointcloud(self, t):
        """Return a point cloud of 3D plume structure at time t."""
        self.solve(t=t, overwrite=True)
        area1 = self.a(t=t)
        area2 = self.b(t=t)
        theta = self.theta(t=t)
        x_loc = self.x_disp(t=t)
        z_loc = self.z_disp(t=t)  # put in global coordinates
        heading = self.heading(t=t)
        current = self.currents(None, t=t)

        pts_all = None
        for i, (a, b, th, x, z) in enumerate(
                zip(area1, area2, theta, x_loc, z_loc)):
            pts = draw_ellipse(a, b)
            pts = to_homogenous(pts[0], pts[1])
            T = translation(x * np.cos(heading) + self.loc[0],
                            x * np.sin(heading) + self.loc[1], z)
            R = xrotation(th / 180. * np.pi)
            Rz = zrotation(heading)

            # Apply affine transformation to points
            pts_p = pts @ R @ Rz @ T
            if pts_all is None:
                pts_all = pts_p
            else:
                pts_all = np.vstack([pts_all, pts_p])

        return pts_all


class Multiplume(CrossflowMTT):
    """Creates a multiplume object from several base plumes."""

    def __init__(self, plume_models):
        """Given a list of plume models, create sampling interfaces.

        Args:
            plume_models (list[plume classes]): list of instantiated models
        """
        self.NAME = f"Multiplume{random_str()}"
        self.models = plume_models
        self._gp_model = None
        ext = self.models[0].extent
        # TODO: put this back in, but check attributes
        # for i in range(1, len(self.models)):
        #     assert(self.models[i].extent == ext)
        self.extent = ext
        self.origins = [m.loc for m in plume_models]

        # Give each multiplume model it's own name
        for i, m in enumerate(self.models):
            m.NAME = f"{self.NAME},{m.NAME},{i}"

    def solve(self, t):
        """Given class parameter settings, compute the spatial solution.

        Args:
            t (float): the global time
            overwrite (bool): if True, will overwrite existing cache.
        """
        # Make sure all models are seeded appropriately
        for model in self.models:
            model.solve(t)

    def write_cache(self, tvec, xrange=None, yrange=None, zrange=None,
                    xres=None, yres=None, zres=None, overwrite=True):
        """Access individual models and set caches.

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
        for i, m in enumerate(self.models):
            m.write_cache(
                tvec=tvec,
                xrange=xrange,
                yrange=yrange,
                zrange=zrange,
                xres=xres,
                yres=yres,
                zres=zres,
                overwrite=overwrite)

    def write_gp_cache(self, tvec, xrange=None, yrange=None,
                       zrange=None, xres=None, yres=None,
                       zres=None, overwrite=True,
                       seed_kernel=None, visualize=False):
        """Access individual models and set caches.

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
        for i, m in enumerate(self.models):
            m.write_gp_cache(
                tvec=tvec,
                xrange=xrange,
                yrange=yrange,
                zrange=zrange,
                xres=xres,
                yres=yres,
                zres=zres,
                overwrite=overwrite,
                seed_kernel=seed_kernel,
                visualize=visualize)

    def _json_stats(self):
        """Return JSON dicts of model information."""
        model_dicts = []
        for m in self.models:
            model_dicts.append(m._json_stats())
        return model_dicts

    def get_value(self, t, loc, return_all=False, from_cache=False,
                  cache_interp="gp"):
        """Get a deterministic prediction of the model output t and loc.

        Args:
            t (float): the global time
            loc (tuple[float]) or list(tuple[float]): a location or
                list of locations, in xyz
            return_all (bool): whether to return more than prob
            from_cache (bool): if True, return nearest value from cache
            cache_interp (str): Either "gp" or "lookup", for Gaussian
                process or look-up table interpolation.

        Returns: tuple(float) plume probability and/or V, S, T
            (probability must be returned as the first element of the tuple)
        """
        model_probs = []
        for i, m in enumerate(self.models):
            # TODO: Why are we scaling here?
            # Scale for later
            out = m.get_value(t, loc, from_cache=from_cache,
                              cache_interp=cache_interp) * 1e-5
            model_probs.append(out)

        model_combos = []
        for r in range(len(model_probs) + 1):
            combinations_object = itertools.combinations(model_probs, r)
            combinations_list = list(combinations_object)
            model_combos += combinations_list

        prob = np.zeros_like(out)
        for c in model_combos:
            if len(c) > 0 and len(c) % 2 == 1:
                prob = prob + np.prod(c, axis=0)
            elif len(c) > 0 and len(c) % 2 == 0:
                prob = prob - np.prod(c, axis=0)
            else:
                pass

        return prob / 1e-5

    def get_snapshot(self, t, z=None, xrange=None, yrange=None, xres=None,
                     yres=None, return_all=False, from_cache=False):
        """Get a deterministic prediction of full state at time t.

        Args:
            t (float): the global time
            z (None or list of floats): heights for which to generate snapshots
            xrange (tuple[float]): the min x and max x to interpolate over
            yrange (tuple[float]): the min y and max y to interpolate over
            xres (int): discretization for x-axis
            yres (int): discretization for y-axis
            return_all (bool): whether to return prob or all values
            from_cache (bool): if True, return nearest value from cache

        Returns: z x nx x ny array of P and/or V, S, T
            (probability must be returned as the first element of the tuple)
        """
        mod_probs = []
        for i, m in enumerate(self.models):
            snapshot = m.get_snapshot(
                t,
                z=z,
                xrange=xrange,
                yrange=yrange,
                xres=xres,
                yres=yres,
                from_cache=from_cache)
            mod_probs.append(snapshot)

        model_combos = []
        for r in range(len(mod_probs) + 1):
            combinations_object = itertools.combinations(mod_probs, r)
            combinations_list = list(combinations_object)
            model_combos += combinations_list

        prob = np.zeros_like(snapshot)
        for c in model_combos:
            if len(c) > 0 and len(c) % 2 == 1:
                prob = prob + np.prod(c, axis=0)
            elif len(c) > 0 and len(c) % 2 == 0:
                prob = prob - np.prod(c, axis=0)
            else:
                pass

        return prob

    def get_maxima(self, t, z=None, xrange=None, yrange=None, xres=None,
                   yres=None, from_cache=False):
        """Get a deterministic prediction of maximum environment value at time t.

        Args:
            t (float): the global time

        Returns: tuple[float] maximum location
        """
        # Get snapshot
        locs = []
        for i, m in enumerate(self.models):
            locs.append(m.get_maxima(t, z, xrange, yrange, xres, yres,
                                     from_cache=from_cache))

        # TODO: how to deal with maxima for mulitple plumes? Average location?
        return np.array(locs).mean()

    def get_pointcloud(self, t):
        """Return a point cloud of 3D plume structure at time t."""
        pts_all = None
        for i, m in enumerate(self.models):
            pts_p = m.get_pointcloud(t)
            if pts_all is None:
                pts_all = pts_p
            else:
                pts_all = np.vstack([pts_all, pts_p])
        return pts_all
