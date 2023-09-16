"""Populates an advective/diffusive layer with particles generated from a PHUMES model"""

import os
import pandas as pd
import utm
import numpy as np
import matplotlib.pyplot as plt
import datetime as datetime
from scipy import interpolate

from fumes.environment.mtt import CrossflowMTT, Multiplume
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S
from fumes.environment.extent import Extent



# Globals
REFERENCE = (-129.094, 47.960, -2125.)
RX, RY, RN, RL = utm.from_latlon(REFERENCE[1], REFERENCE[0])
HR_COORDS = (-129.0894, 47.9666, -2125.) # HR
MEF_COORDS = (-129.0981, 47.9487, -2125.) # MEF
CHIMNEYS = [HR_COORDS, MEF_COORDS]
locs = []
for sc in CHIMNEYS:
    cx, cy, _, _ = utm.from_latlon(sc[1], sc[0])
    locs.append((cx - RX, cy - RY, 0))

# Parameters
z = np.linspace(0, 200, 100)  # height to integrate over
s = np.linspace(0, 500, 100)  # distance to integrate over
tprof = pacific_sp_T  # function that describes background temp
sprof = pacific_sp_S  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T

times = np.linspace(0, 24 * 3600, 24 * 3600+1)
query_times = [0, 6, 12, 18, 24]  # , 6, 12]  # in hours


cdata = pd.read_csv(os.path.join(os.getenv("JDFR_OUTPUT"), "sentry/ocn_data_dive686.csv"))
cdata.Datetime = pd.to_datetime(cdata.Datetime)
datetime_ref = datetime.datetime(year=2023, month=8, day=27, hour=1, minute=0, second=0, tzinfo=datetime.timezone.utc)
cdata = cdata[cdata.Datetime > datetime_ref]
cdata.loc[:, "t_zeroed"] = cdata.t - cdata.t.values[0]


northingfunc = interpolate.interp1d(cdata.t_zeroed, cdata.true_vnorth, bounds_error=False, fill_value="extrapolate")
eastingfunc = interpolate.interp1d(cdata.t_zeroed, cdata.true_veast, bounds_error=False, fill_value="extrapolate")

def heading(T):
    return np.arctan2(northingfunc(T), eastingfunc(T))
def magnitude(x, T):
    return np.sqrt(northingfunc(T)**2 + eastingfunc(T)**2)


fig, ax = plt.subplots(2, 1)
ax[0].plot(times, heading(times))
ax[1].plot(times, magnitude(None, times))
plt.show()

loc = (0., 0., 0.)

v0s = [0.1, 0.1]  # source exit velocity
a0s = [1.0, 1.0]  # source area
s0s = [sprof(0), sprof(0)]  # source salinity
t0s = [350., 320.]  # source temperature
rho0s = [eos_rho(t0s[0], s0s[0]), eos_rho(t0s[1], s0s[1])]  # source density

lam = 1.0  # for crossflow, major-minor axis ratio
entrainment = [0.12, 0.2]  # entrainment coeffs

extent = Extent(xrange=(-2000., 2000.),
                xres=200,
                yrange=(-2000., 2000.),
                yres=200,
                zrange=(0, 200),
                zres=20,
                global_origin=(0, 0, 0))

#####
#Helpers
#####
def create_crossflow_world(locs, extent, s, tprof, sprof, rhoprof, curfunc, headfunc, v0s, a0s, s0s, t0s, rho0s, lam, entrainment):
    """Creates a list of crossflow worlds."""
    envs = []
    for loc, v0, a0, s0, t0, rho0 in zip(locs, v0s, a0s, s0s, t0s, rho0s):
        w = CrossflowMTT(plume_loc=loc,
                         extent=extent,
                         s=s,
                         tprof=tprof,
                         sprof=sprof,
                         rhoprof=rhoprof,
                         curfunc=curfunc,
                         headfunc=headfunc,
                         vex=v0,
                         area=a0,
                         salt=s0,
                         temp=t0,
                         density=rho0,
                         lam=lam,
                         entrainment=entrainment)
        envs.append(w)
    return envs

if __name__ == "__main__":
    # define the environments
    envs = create_crossflow_world(locs=locs,
                                extent=extent,
                                s=s,
                                tprof=tprof,
                                sprof=sprof,
                                rhoprof=rhoprof,
                                curfunc=magnitude,
                                headfunc=heading,
                                v0s=v0s,
                                a0s=a0s,
                                s0s=s0s,
                                t0s=t0s,
                                rho0s=rho0s,
                                lam=lam,
                                entrainment=entrainment)
    multiplume = Multiplume(envs)

    # Get a plume intersection
    y = np.linspace(-50, 50, 100)
    x = np.zeros_like(y)
    height = np.zeros(y.shape[0] * x.shape[0])
    pq = multiplume.get_value(t=10, loc=(x, y, height), from_cache=False)
    plt.plot(y, pq)
    plt.xlabel('Y-coordinate')
    plt.ylabel('Plume-State')
    plt.title('Environment Slice at X=0')
    plt.show()

    # Track the intersection of the plume centerline at fixed height
    query_height = 125
    for t in query_times:
        # Draw a vertical slice of the plume envelope
        envelopes = multiplume.envelope(t=t * 3600.)
        for envelope in envelopes:
            le, cl, re = envelope
            intersections = np.fabs(cl[1] - np.ones_like(cl[1]) * 125.)
            intersection_idx = intersections < 5
            intersection_xvals = cl[0][intersection_idx]
            intersection_x_avg = np.nanmean(intersection_xvals)  # distance from true source
            world_coord = (np.cos(heading(t*3600.)) * intersection_x_avg, np.sin(heading(t*3600.)) * intersection_x_avg)
            print(world_coord)

    # Get a birds-eye snapshot of plume probabilities
    fig, ax = plt.subplots(len(query_times), 2, sharex=True, sharey=True)
    if len(query_times) > 1:
        for i, qt in enumerate(query_times):
            ps = multiplume.get_snapshot(t=qt * 60. * 60., z=[20, 125])
            vmin = np.nanpercentile(ps.flatten(), 1)
            vmax = np.nanpercentile(ps.flatten(), 99)
            ax[i, 0].imshow(ps[0], origin="lower", extent=(-2000, 2000, -2000, 2000), vmin=vmin, vmax=vmax)
            ax[i, 0].set_ylabel('Y-coordinate')
            ax[i, 0].set_title("Z=20m, All Plumes")
            ax[i, 1].imshow(ps[1], origin="lower", extent=(-2000, 2000, -2000, 2000), vmin=vmin, vmax=vmax)
            ax[i, 1].set_ylabel('Y-coordinate')
            ax[i, 1].set_title("Z=125m, All Plumes")
    else:
        ps = multiplume.get_snapshot(t=query_times[0] * 60. * 60., z=[20, 125])
        vmin = np.nanpercentile(ps.flatten(), 1)
        vmax = np.nanpercentile(ps.flatten(), 99)
        ax[0].imshow(ps[0], origin="lower", extent=(-2000, 2000, -2000, 2000), vmin=vmin, vmax=vmax)
        ax[0].set_ylabel('Y-coordinate')
        ax[0].set_title("Z=20m, All Plumes")
        ax[1].imshow(ps[1], origin="lower", extent=(-2000, 2000, -2000, 2000), vmin=vmin, vmax=vmax)
        ax[1].set_ylabel('Y-coordinate')
        ax[1].set_title("Z=125m, All Plumes")
    plt.show()
