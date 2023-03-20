"""Utilities for environment files."""
import os
import sys
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
import contextlib
import os
import sys
import random
import pdb
import string

import gpytorch as gpy
import torch

from fumes.utils import data_home, output_home


def random_str(n=4):
    return ''.join(
        random.choice(string.ascii_uppercase + string.digits) for _ in range(n))


# A circular x and y coordinate over the course of 12 hours,
# offset from the origin


def xcoord_circle(t):
    return 100 * np.cos(t * (2 * np.pi) / (12 * 60 * 60)) + 300


def ycoord_circle(t):
    return -100 * np.sin(t * (2 * np.pi) / (12 * 60 * 60)) + 300


# A target area, lengthscale, and angle that oscillate
def l_oscillate(t, l):
    return (l / 2.0) * np.cos(t * (2 * np.pi) / (12 * 60 * 60)) + l


def A_oscillate(t, A):
    return A * np.sin(t * (2 * np.pi) / (12 * 60 * 60))


def theta_oscillate(t):
    return (2 * np.pi) * np.sin(t * (2 * np.pi) / (48 * 60 * 60))

# A sinusoidal current and heading function


def curfunc_test(x, t):
    # return 0.2 * np.cos(2 * np.pi * t / (24 * 3600.))
    # return 0.2 * np.abs()
    # return abs(0.2 * np.cos(2 * np.pi * t / (24 * 3600.)))
    # hrs = 6
    # t = t % (hrs*3600.)
    # return 0.2*abs(-1 + (t / (hrs * 3600))*2.)
    return np.ones_like(t) * 0.1


def headfunc_test(t):
    # angle = 80.0
    # return np.ones_like(t) * (angle + 90.0) * np.pi / 180.0
    # start = -45.0 # add 90 to convert to north as zero convention
    # end = 170.0 # add 90 to convert to north as zero convention

    # angle = 214.*np.sin(2 * np.pi * t / (12 * 3600.)) + 62.0
    angle = 110.
    return np.ones_like(t) * angle / 180. * np.pi

    # t = t % (12.*3600.)
    # return start + abs((t / (12. * 3600))*(end.)
    # return np.ones_like(t) * (start + (t / (12*3600)) * (end - start)) * np.pi / 180.0

# A sinusoidal current and heading function


def curfunc(x, t):
    # return 0.2 * np.cos(2 * np.pi * t / (24 * 3600.))
    # return 0.2 * np.abs()
    # return abs(0.2 * np.cos(2 * np.pi * t / (24 * 3600.)))
    # hrs = 6
    # t = t % (hrs*3600.)
    # return 0.2*abs(-1 + (t / (hrs * 3600))*2.)
    mag = 0.04 * np.cos(2 * np.pi * t / (12 * 3600.)) + 0.08
    return mag
    # return mag / 180. * np.pi
    # return mag / 180. * np.pi


def headfunc(t):
    angle = 50. * np.cos(2 * np.pi * t / (24 * 3600.)) + 45.0
    return angle / 180. * np.pi


def headfunc_fast(t):
    angle = 50. * np.cos(2 * np.pi * (2 * t) / (24 * 3600.)) + 45.0
    return angle / 180. * np.pi


def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def get_closest_point(qpx, qpy, lpx, lpy):
    distance = cdist(np.column_stack((qpx, qpy)), np.column_stack((lpx, lpy)))
    idx = np.argpartition(distance, 2, axis=1)
    nx1, ny1, nx2, ny2 = (lpx[idx[:, 0]],
                          lpy[idx[:, 0]],
                          lpx[idx[:, 1]],
                          lpy[idx[:, 1]])
    dx, dy = nx2 - nx1, ny2 - ny1
    det = dx * dx + dy * dy
    a = (dy * (qpy - ny1) + dx * (qpx - nx1)) / det
    return nx1 + a * dx, ny1 + a * dy

# Standard profiles for Atlantic and Pacific salinity and temperature


def atlantic_sp_T(z):
    """Function that produces the linear atlantic standard T profile.

    From K. G. Speer and P. A. Rona. "A model of an Atlantic and Pacific
    hydrothermal plume," Journal of Geophysical Research: Oceans, vol.94,
    no. C5, pp.6213-6220, 1989.
    """
    tprof = 2.35 + 4.0 * 10**(-4) * z
    return tprof


def atlantic_sp_S(z):
    """Function that produces the linear atlantic standard S profile.

    From K. G. Speer and P. A. Rona. "A model of an Atlantic and Pacific
    hydrothermal plume," Journal of Geophysical Research: Oceans, vol.94,
    no. C5, pp.6213-6220, 1989.
    """
    sprof = 35.923 + 4.0 * 10**(-5) * z
    return sprof


def pacific_sp_T(z):
    """Function that produces the linear pacific standard T profile.

    From K. G. Speer and P. A. Rona. "A model of an Atlantic and Pacific
    hydrothermal plume," Journal of Geophysical Research: Oceans, vol.94,
    no. C5, pp.6213-6220, 1989.
    """
    tprof = 1.80 + 10**(-3) * z
    return tprof


def pacific_sp_S(z):
    """Function that produces the linear pacific standard S profile.

    From K. G. Speer and P. A. Rona. "A model of an Atlantic and Pacific
    hydrothermal plume," Journal of Geophysical Research: Oceans, vol.94,
    no. C5, pp.6213-6220, 1989.
    """
    sprof = 34.608 - 10**(-4) * z
    return sprof


# Equation of state to compute density from S, T
def eos_rho(T, S):
    """Equation of state for density.

    From K. G. Speer and P. A. Rona. "A model of an Atlantic and Pacific
    hydrothermal plume," Journal of Geophysical Research: Oceans, vol.94,
    no. C5, pp.6213-6220, 1989.
    """
    temp = 2.13 * 10**(-4) * (T - 2.0)
    salt = 7.5 * 10**(-4) * (S - 34.89)
    return 1.041548 - temp + salt  # EOS


# Numerical models for various plume environments
def bent_speer_rona(v, z, t, rho_o, sbar, tbar, rhobar, ubar,
                    E=0.255, g=-9.81):
    """The Speer and Rona numerical model for plumes with crossflow.

    Adds crossflow based on simplification made by Middleton 1985
    in which dz/dx = V/U is used as a substitution in volume,
    momentum, and buoyancy. Here, we solve Speer and Rona directly
    and just add the height relationship as a seperate equation.
    This means that area will not be influenced by crossflow; it will just
    look like horizontally advecting circles.

    Developed from equations in:
    - K. G. Speer and P. A. Rona. "A model of
      an Atlantic and Pacific hydrothermal plume," Journal of
      Geophysical Research: Oceans, vol.94, no. C5, pp.6213-6220, 1989.
    - J. H. Middleton, "The rise of forced plumes in a stably
      stratified cross-flow," Boundary-Layer Meterology, vol. 36, no. 1,
      pp. 187-199, 1986.
    - B. Morton, G. I. Taylor, and J. S. Turner, "Turbulent
      graviational convection from maintained and instantaneous sources,"
      Proceedings of the Royal Society of London. Series A. Mathematical
      and Physical Sciences, vol. 234, no. 1196, pp. 1-23, 1956.
    """
    V, A, S, T, X = v  # variables in the model
    rho = eos_rho(T, S)
    backT = tbar(z)
    backS = sbar(z)
    rhob = rhobar(backT, backS)
    dadz = 4 * E * np.sqrt(A) - (2 * g * A) / (rho_o * V**2) * (rho - rhob)
    dvdz = 2 * g / (2 * rho_o * V) * (rho - rhob) - dadz * V / (2 * A)
    dsdz = 2 * sbar(z) * E / np.sqrt(A) - dadz * S / A - dvdz * S / V
    dtdz = 2 * tbar(z) * E / np.sqrt(A) - dadz * T / A - dvdz * T / V
    dxdz = ubar(z, t) / V  # this is the key difference with Speer/Rona
    return [dvdz, dadz, dsdz, dtdz, dxdz]


def speer_rona(v, z, rho_o, sbar, tbar, rhobar, E=0.255, g=-9.81):
    """The Speer and Rona numerical model for plumes.

    Developed from equations in:
    - K. G. Speer and P. A. Rona. "A model of
      an Atlantic and Pacific hydrothermal plume," Journal of
      Geophysical Research: Oceans, vol.94, no. C5, pp.6213-6220, 1989.
    - B. Morton, G. I. Taylor, and J. S. Turner, "Turbulent
      graviational convection from maintained and instantaneous sources,"
      Proceedings of the Royal Society of London. Series A. Mathematical
      and Physical Sciences, vol. 234, no. 1196, pp. 1-23, 1956.
    """
    V, A, S, T = v  # variables in the model
    rho = eos_rho(T, S)
    backT = tbar(z)
    backS = sbar(z)
    rhob = rhobar(backT, backS)
    dadz = 4 * E * np.sqrt(A) - (2 * g * A) / (rho_o * V**2) * (rho - rhob)
    dvdz = 2 * g / (2 * rho_o * V) * (rho - rhob) - dadz * V / (2 * A)
    dsdz = 2 * sbar(z) * E / np.sqrt(A) - dadz * S / A - dvdz * S / V
    dtdz = 2 * tbar(z) * E / np.sqrt(A) - dadz * T / A - dvdz * T / V
    return [dvdz, dadz, dsdz, dtdz]


def xu_diorio():
    """WIP function in case of interest."""
    raise NotImplementedError("This is a WIP function, see code for details.")
    Ua = 0.1
    rho = 1000
    E = 0.255
    g = -9.81
    Cd = 0.1
    rho_o = 1000
    drhods = 0.5

    s = symbols('s')
    b, u, t, m = symbols('b u t m', cls=Function)

    # mass functions
    expr = b(s)**2 * (rho * (0.432 * u(s) + Ua * cos(t(s))) -
                      m(s) * (0.540 * Ua * cos(t(s) + 0.285 * u(s))))
    dmdt = expr.diff(s)
    mass = simplify(Eq(dmdt, np.sqrt(2) * b(s) * rho * E))

    # vert momentum
    e1 = rho * (Ua**2 * cos(t(s))**2 + 0.254 * u(s)**2 + 0.865 * Ua * u(s) *
                cos(t(s)))
    e2 = m(s) * (0.540 * Ua**2 * cos(t(s))**2 + 0.57 * Ua * u(s) *
                 cos(t(s)) + 0.185 * u(s)**2)
    e3 = 0.540 * b(s)**2 * g * m(s) - np.sqrt(2) / 2 * b(s) * \
        rho * Cd * (Ua * sin(t(s)))**2 * cos(t(s))
    expr_vm = b(s)**2 * sin(t(s)) * (e1 - e2)
    dvmds = expr_vm.diff(s)
    vm = simplify(Eq(dvmds, e3))

    # horz momentum
    e4 = np.sqrt(2) * b(s) * rho * Ua * E + np.sqrt(2) / 2 * \
        b(s) * rho * Cd * (Ua * sin(t(s)))**2 * sin(t(s))
    expr_hm = b(s)**2 * cos(t(s)) * (e1 - e2)
    dhmds = expr_hm.diff(s)
    hm = simplify(Eq(dhmds, e4))

    # dens diff
    e5 = rho * (0.285 * u(s) + 0.540 * Ua * cos(t(s)))
    e6 = m(s) * (0.338 * Ua * cos(t(s)) + 0.208 * u(s))
    expr_dd = m(s) * b(s)**2 / rho * (e5 - e6)
    ddds = expr_dd.diff(s)
    e7 = rho * (0.432 * u(s) + Ua * cos(t(s)))
    e8 = m(s) * (0.285 * u(s) + 0.540 * Ua * cos(t(s)))
    e9 = 1 / rho_o * drhods * b(s)**2 * (e7 - e8)
    dd = simplify(Eq(ddds, e9))

    eqs = [mass, vm, hm, dd]
    print(eqs)
    print('------')
    sol = dsolve_system(eqs, funcs=[b(s), u(s), t(s), m(s)], t=s)
    print(sol)


def tohidi_kaye(v, s, t, rho_o, ubar, tbar, sbar, rhobar,
                lam=1.0, E=[0.255, 0], g=-9.81, zoff=0):
    """The Tohidi-Kaye crossflow model with empirical adjustments.

    Developed from equations in:
    - A. Tohidi and N. B. Kaye, "Highly buoyant bent-over plumes
      in a boundary layer," Atmospheric Environment, vol. 131,
      pp. 97-114, 2016.
    - G. Xu and D. Di Iorio, "Deep sea hydrothermal plumes and their
      interactions with oscillatory flows," Geochemistry, Geophysics,
      Geosystems, vol. 13, no. 9, 2012.
    """
    Q, M, F, T, x, z = v  # breakdown the interesting variables

    dz = 1.0  # arbitrarily small unit to compute drhob/dz
    backT = tbar(z + zoff)
    backS = sbar(z + zoff)
    rhob = rhobar(backT, backS)
    drhob = (rhobar(tbar(z + zoff + dz), sbar(z + zoff + dz)) - rhob) / dz
    temp = F / (Q * -g * 10**(-4)) + backT
    rho = eos_rho(temp, backS)
    N2 = g / rhob * drhob
    alpha = E[0]
    beta = E[1]
    uval = ubar(z + zoff, t)
    entrainment = alpha * np.fabs(M / Q - uval * np.cos(T)) + beta * \
        np.fabs(uval * np.sin(T))

    dqds = Q * np.sqrt(2 * (1 + lam**2) / (M * lam)) * entrainment
    dmds = uval * np.cos(T) * dqds + F * Q / M * np.sin(T)
    dtds = (F * Q / M * np.cos(T) - uval * np.sin(T) * dqds) / M
    dfds = -Q * N2 * np.sin(T)
    dxds = np.cos(T)
    dzds = np.sin(T)

    return [dqds, dmds, dfds, dtds, dxds, dzds]


def rotgauss(A, x0, y0, x, y, a, b, theta):
    """Returns a Gaussian function with a rotation."""
    cx = x0 * np.cos(theta) - y0 * np.sin(theta)
    cy = x0 * np.sin(theta) + y0 * np.cos(theta)

    xp = x * np.cos(theta) - y * np.sin(theta)
    yp = x * np.sin(theta) + y * np.cos(theta)
    return A * np.exp(-(((cx - xp) / a)**2 + ((cy - yp) / b)**2) / 2.0)


def ncdump(nc_fid, verb=True):
    '''
    ncdump outputs dimensions, variables and their attribute information.
    The information is similar to that of NCAR's ncdump utility.
    ncdump requires a valid instance of Dataset.
    From: schubert.atmos.colostate.edu/~cslocum/netcdf_example.html

    Parameters
    ----------
    nc_fid : netCDF4.Dataset
        A netCDF4 dateset object
    verb : Boolean
        whether or not nc_attrs, nc_dims, and nc_vars are printed

    Returns
    -------
    nc_attrs : list
        A Python list of the NetCDF file global attributes
    nc_dims : list
        A Python list of the NetCDF file dimensions
    nc_vars : list
        A Python list of the NetCDF file variables
    '''
    def print_ncattr(key):
        """
        Prints the NetCDF file attributes for a given key

        Parameters
        ----------
        key : unicode
            a valid netCDF4.Dataset.variables key
        """
        try:
            print("\t\ttype:", repr(nc_fid.variables[key].dtype))
            for ncattr in nc_fid.variables[key].ncattrs():
                print('\t\t%s:' % ncattr,
                      repr(nc_fid.variables[key].getncattr(ncattr)))
        except KeyError:
            print("\t\tWARNING: %s does not contain variable attributes" % key)

    # NetCDF global attributes
    nc_attrs = nc_fid.ncattrs()
    if verb:
        print("NetCDF Global Attributes:")
        for nc_attr in nc_attrs:
            print('\t%s:' % nc_attr, repr(nc_fid.getncattr(nc_attr)))
    nc_dims = [dim for dim in nc_fid.dimensions]  # list of nc dimensions
    # Dimension shape information.
    if verb:
        print("NetCDF dimension information:")
        for dim in nc_dims:
            print("\tName:", dim)
            print("\t\tsize:", len(nc_fid.dimensions[dim]))
            print_ncattr(dim)
    # Variable information.
    nc_vars = [var for var in nc_fid.variables]  # list of nc variables
    if verb:
        print("NetCDF variable information:")
        for var in nc_vars:
            if var not in nc_dims:
                print('\tName:', var)
                print("\t\tdimensions:", nc_fid.variables[var].dimensions)
                print("\t\tsize:", nc_fid.variables[var].size)
                print_ncattr(var)
    return nc_attrs, nc_dims, nc_vars


def fileno(file_or_fd):
    fd = getattr(file_or_fd, 'fileno', lambda: file_or_fd)()
    if not isinstance(fd, int):
        raise ValueError("Expected a file (`.fileno()`) or a file descriptor")
    return fd


@contextlib.contextmanager
def stdout_redirected(to=os.devnull, stdout=None):
    """
    https://stackoverflow.com/a/22434262/190597 (J.F. Sebastian)
    """
    if stdout is None:
        stdout = sys.stdout

    stdout_fd = fileno(stdout)
    # copy stdout_fd before it is overwritten
    # NOTE: `copied` is inheritable on Windows when duplicating a
    # standard stream
    with os.fdopen(os.dup(stdout_fd), 'wb') as copied:
        stdout.flush()  # flush library buffers that dup2 knows nothing about
        try:
            os.dup2(fileno(to), stdout_fd)  # $ exec >&to
        except ValueError:  # filename
            with open(to, 'wb') as to_file:
                os.dup2(to_file.fileno(), stdout_fd)  # $ exec > to
        try:
            yield stdout  # allow code to be run with the redirected stdout
        finally:
            # restore stdout to its previous value
            # NOTE: dup2 makes stdout_fd inheritable unconditionally
            stdout.flush()
            os.dup2(copied.fileno(), stdout_fd)  # $ exec >&copied

# TODO: we'd like this to be in the model utils;
# figure out the circular dependence

def load_bathy_by_coord(minlat, maxlat, minlon, maxlon):
    bathy_files = []
    if minlat >= 9.932:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.912:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.902:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.892:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.882:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.872:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.862:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.852:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.842:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.832:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.822:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.812:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.802:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.792:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.782:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_15.txt"))
        if maxlat >= 9.792:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.772:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_16.txt"))
        if maxlat >= 9.782:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_15.txt"))
        if maxlat >= 9.792:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.762:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_17.txt"))
        if maxlat >= 9.772:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_16.txt"))
        if maxlat >= 9.782:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_15.txt"))
        if maxlat >= 9.792:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.752:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_18.txt"))
        if maxlat >= 9.762:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_17.txt"))
        if maxlat >= 9.772:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_16.txt"))
        if maxlat >= 9.782:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_15.txt"))
        if maxlat >= 9.792:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.742:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_19.txt"))
        if maxlat >= 9.752:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_18.txt"))
        if maxlat >= 9.762:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_17.txt"))
        if maxlat >= 9.772:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_16.txt"))
        if maxlat >= 9.782:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_15.txt"))
        if maxlat >= 9.792:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.732:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_20.txt"))
        if maxlat >= 9.742:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_19.txt"))
        if maxlat >= 9.752:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_18.txt"))
        if maxlat >= 9.762:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_17.txt"))
        if maxlat >= 9.772:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_16.txt"))
        if maxlat >= 9.782:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_15.txt"))
        if maxlat >= 9.792:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    elif minlat >= 9.721:
        bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_21.txt"))
        if maxlat >= 9.732:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_20.txt"))
        if maxlat >= 9.742:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_19.txt"))
        if maxlat >= 9.752:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_18.txt"))
        if maxlat >= 9.762:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_17.txt"))
        if maxlat >= 9.772:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_16.txt"))
        if maxlat >= 9.782:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_15.txt"))
        if maxlat >= 9.792:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_14.txt"))
        if maxlat >= 9.802:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_13.txt"))
        if maxlat >= 9.812:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_12.txt"))
        if maxlat >= 9.822:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_11.txt"))
        if maxlat >= 9.832:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_10.txt"))
        if maxlat >= 9.842:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_9.txt"))
        if maxlat >= 9.852:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_8.txt"))
        if maxlat >= 9.862:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_7.txt"))
        if maxlat >= 9.872:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_6.txt"))
        if maxlat >= 9.882:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_5.txt"))
        if maxlat >= 9.892:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_4.txt"))
        if maxlat >= 9.902:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_3.txt"))
        if maxlat >= 9.912:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_2.txt"))
        if maxlat >= 9.932:
            bathy_files.append(os.path.join(os.getenv("EPR_DATA"), f"bathy/epr_1.txt"))
    else:
        print("Sorry, these coordinates do not match our bathy data!")
    return bathy_files

def get_bathy(lat_min, lat_max, lon_min, lon_max, buffer=0.01, rsamp=0.1):
    """Retrieves bathy data around a given site."""
    # read in the bathy data
    bathy_files = load_bathy_by_coord(lat_min - buffer,
                                        lat_max + buffer,
                                        lon_min - buffer,
                                        lon_max + buffer)
    bathy_dfs = []
    for f in bathy_files:
        print(f)
        temp = pd.read_table(f, names=["lon", "lat", "depth"]).dropna()
        bathy_dfs.append(temp)
    bathy = pd.concat(bathy_dfs, ignore_index=True)
    print(bathy)

    # extract bathy with a decimal degree buffer in each direction
    bathy = bathy[(bathy.lat < lat_max + buffer) & (bathy.lat > lat_min - buffer) &
                    (bathy.lon < lon_max + buffer) & (bathy.lon > lon_min - buffer)]

    # subsample bathy data and return
    return bathy.sample(frac=rsamp, random_state=1)


class ExactGPModel(gpy.models.ExactGP):
    """ Gaussian process regression model. """

    def __init__(self, train_x, train_y, likelihood, num_dims=1, name=None):
        super().__init__(train_x, train_y, likelihood)
        # self.likelihood = gpy.likelihoods.GaussianLikelihood(
        #     noise_constraint=gpy.constraints.LessThan(1e-2))
        # TODO: figure out if constant mean is working?
        # self.mean_module = gpy.means.ConstantMean()
        self.mean_module = gpy.means.ZeroMean()
        self.covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(ard_num_dims=num_dims, has_lengthscale=True) * gpy.kernels.RBFKernel(ard_num_dims=num_dims, has_lengthscale=True) * gpy.kernels.RBFKernel(ard_num_dims=num_dims, has_lengthscale=True))

        if name is None:
            name = "model_state.pth"
        self.model_file = os.path.join(output_home(), f"{name}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModelPeriodic(gpy.models.ExactGP):
    """ Gaussian process regression model. """

    def __init__(self, train_x, train_y, likelihood, num_dims=1, name=None):
        super().__init__(train_x, train_y, likelihood)
        # self.likelihood = gpy.likelihoods.GaussianLikelihood(
        #     noise_constraint=gpy.constraints.LessThan(1e-2))
        # TODO: figure out if constant mean is working?
        # self.mean_module = gpy.means.ConstantMean()
        self.mean_module = gpy.means.ZeroMean()
        self.covar_module = gpy.kernels.ScaleKernel(
            gpy.kernels.RBFKernel(ard_num_dims=num_dims, has_lengthscale=True) * gpy.kernels.RBFKernel(ard_num_dims=num_dims, has_lengthscale=True) * gpy.kernels.PeriodicKernel(ard_num_dims=num_dims))

        if name is None:
            name = "model_state.pth"
        self.model_file = os.path.join(output_home(), f"{name}")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


class StandardApproximateGP(gpy.models.ApproximateGP):
    """Approximate GP regression model."""

    def __init__(self, inducing_points, name=None):

        variational_distribution = \
            gpy.variational.CholeskyVariationalDistribution(inducing_points.size(-2))
        variational_strategy = gpy.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpy.means.ConstantMean()
        self.covar_module = gpy.kernels.ScaleKernel(gpy.kernels.RBFKernel())
        if name is None:
            name = "model_state"
        self.model_file = os.path.join(data_home(), f"{name}.pth")

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpy.distributions.MultivariateNormal(mean_x, covar_x)


