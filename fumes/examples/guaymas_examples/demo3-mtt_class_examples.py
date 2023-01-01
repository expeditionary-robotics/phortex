"""Instantiates all of the MTT environments in example."""

from re import X
import matplotlib.pyplot as plt
import numpy as np
from fumes.environment.mtt import Multiplume, StationaryMTT, CrossflowMTT
from fumes.environment.utils import eos_rho, pacific_sp_T, pacific_sp_S, \
    curfunc, headfunc
from fumes.environment.extent import Extent

# Parameters
z = np.linspace(0, 200, 100)  # height to integrate over
tprof = pacific_sp_T  # function that describes background temp
sprof = pacific_sp_S  # function that describes background salt
rhoprof = eos_rho  # function that computes density as func of S, T


loc = (0., 0., 0.)

v0 = 0.2  # source exit velocity
a0 = 0.5  # source area
s0 = 34.608  # source salinity
t0 = 300.  # source temperature
rho0 = eos_rho(t0, s0)  # source density

lam = 1.0  # for crossflow, major-minor axis ratio
entrainment = [0.12, 0.2]  # entrainment coeffs

# Choose examples to run
RUN_STATIONARY = True
RUN_CROSSFLOW = True
RUN_MULTIPLUME = True

###################################################
# Create StationaryMTT and plot environment outputs
###################################################
if RUN_STATIONARY is True:
    extent = Extent(xrange=(-500., 500.),
                    xres=200,
                    yrange=(-500., 500.),
                    yres=200,
                    zrange=(0, 50),
                    zres=10,
                    global_origin=loc)
    mtt = StationaryMTT(plume_loc=loc,
                        extent=extent,
                        z=z,
                        tprof=tprof,
                        sprof=sprof,
                        rhoprof=rhoprof,
                        vex=v0,
                        area=a0,
                        salt=s0,
                        temp=t0,
                        density=rho0)

    # First, compare plume and background waters
    output = ['Velocity', 'Area', 'Salinity', 'Temperature']
    names = ["vel", "area", "sal", "temp"]
    mtt_z = mtt.z
    compare = [None, None, pacific_sp_S, pacific_sp_T]
    f, ax = plt.subplots(2, 2, sharex=True, figsize=(10, 10))
    ax = ax.flatten()
    for i, (var, title) in enumerate(zip(names, output)):
        ax[i].plot(mtt_z, mtt.get(t=0.0, variable=var), label='Plume')
        if compare[i] is not None:
            ax[i].plot(mtt_z, compare[i](mtt_z), '--', label='Background')
        ax[i].set_xlabel('Altitude (z) (meters)')
        ax[i].set_ylabel(output[i])
        ax[i].set_title(output[i])
        ax[i].legend()
    plt.show()

    # Draw a vertical slice of the plume envelope
    le, cl, re = mtt.envelope(t=0.0)
    plt.plot(np.zeros_like(z), z, label="Centerline")
    plt.plot(le, z, label="Left Extent")
    plt.plot(re, z, label="Right Extent")
    plt.title("Plume Envelope")
    plt.xlabel("X (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    plt.show()

    # Get a plume intersection of points in salinity
    y = np.linspace(-500, 500, 1000)
    x = np.zeros_like(y)
    height = np.ones_like(x) * 100
    _, sq, _, _ = mtt.get_value(t=0.0, loc=(x, y, height), return_all=True, from_cache=False)
    plt.plot(y, sq)
    plt.xlabel('Y-coordinate')
    plt.ylabel('Salinity')
    plt.title('Environment Slice at X=0')
    plt.show()


    # Get a birds-eye snapshot of plume probability
    ps = mtt.get_snapshot(t=0.0, z=[0, 75], from_cache=False)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax[0].imshow(ps[0], origin="lower", extent=(-500, 500, -500, 500))
    ax[0].set_xlabel('X-coordinate')
    ax[0].set_ylabel('Y-coordinate')
    ax[0].set_title("Z=0m")
    ax[1].imshow(ps[1], origin="lower", extent=(-500, 500, -500, 500))
    ax[1].set_xlabel('X-coordinate')
    ax[1].set_ylabel('Y-coordinate')
    ax[1].set_title("Z=75m")
    plt.show()

    # Cache the environment, then read from it
    mtt.write_cache(tvec=[0, 75],
                       xrange=(0, 200),
                       yrange=(0, 200),
                       zrange=(0, 200),
                       xres=100,
                       yres=100,
                       zres=10)

    query_point = (np.linspace(10, 10, 1),
                   np.linspace(10, 10, 1),
                   np.linspace(200, 200, 1))

    print("Value from model: ", mtt.get_value(t=75,
                                              loc=query_point,
                                              from_cache=False))
    print("Value from cache: ", mtt.get_value(t=75,
                                              loc=query_point,
                                              from_cache=True,
                                              cache_interp="lookup"))

    Plive = mtt.get_snapshot(t=75,
                             z=[50],
                             xrange=(0, 200),
                             yrange=(0, 200),
                             xres=100,
                             yres=100,
                             from_cache=False)
    Pcache = mtt.get_snapshot(t=75,
                              z=[50],
                              xrange=(0, 200),
                              yrange=(0, 200),
                              xres=100,
                              yres=100,
                              from_cache=True)
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    minmin = np.nanmin(Plive.flatten())
    maxmax = np.nanmax(Plive.flatten())
    ax[0].imshow(Plive[0], origin="lower", extent=(0, 200, 0, 200),
                 vmin=minmin, vmax=maxmax)
    ax[0].set_title("In-Memory Snapshot")
    ax[1].imshow(Pcache[0], origin="lower", extent=(0, 200, 0, 200),
                 vmin=minmin, vmax=maxmax)
    ax[1].set_title("Cached Snapshot")
    plt.show()

###################################################
# Create CrossflowMTT and plot environment outputs
###################################################
if RUN_CROSSFLOW is True:
    s = np.linspace(0, 1000, 100)
    extent = Extent(xrange=(-200., 200.),
                    xres=100,
                    yrange=(-200., 200.),
                    yres=100,
                    zrange=(0, 50),
                    zres=10,
                    global_origin=loc)
    mtt = CrossflowMTT(plume_loc=loc,
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

    # First, look at all of the profiles with respect to depth
    output = ['Volume Transport', 'Momentum Transport', 'Buoyancy Transport',
              'Angle', 'X', 'Z']
    names = ["q", "m", "f", "theta", "x_disp", "z_disp"]

    f, ax = plt.subplots(3, 2, sharex=True, figsize=(15, 10))
    ax = ax.flatten()
    for i, (var, title) in enumerate(zip(names, output)):
        ax[i].plot(mtt.z_disp(t=0.0), mtt.get(t=0.0, variable=var), label='Plume')
        ax[i].set_xlabel('Altitude (z) (meters)')
        ax[i].set_ylabel(output[i])
        ax[i].set_title(output[i])
        ax[i].legend()
    plt.show()

    # Draw a vertical slice of the plume envelope
    le, cl, re = mtt.envelope(t=0.0)

    plt.plot(*cl, label="Centerline")
    plt.plot(*le, label="Left Extent")
    plt.plot(*re, label="Right Extent")
    plt.title("Plume Envelope")
    plt.xlabel("X in crossflow direction (meters)")
    plt.ylabel("Z (meters)")
    plt.legend()
    plt.show()

    # Get a plume intersection
    y = np.linspace(-10, 10, 1000)
    x = np.zeros_like(y)
    height = np.zeros(y.shape[0] * x.shape[0])
    pq = mtt.get_value(t=10, loc=(x, y, height), from_cache=False)
    plt.plot(y, pq)
    plt.xlabel('Y-coordinate')
    plt.ylabel('Plume-State')
    plt.title('Environment Slice at X=0')
    plt.show()

    # Get a birds-eye snapshot of plume probability
    ps = mtt.get_snapshot(t=0, z=[10, 60], from_cache=False)
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    minmin = np.nanmin(ps.flatten())
    maxmax = np.nanmax(ps.flatten())
    ax[0].imshow(ps[0], origin="lower", extent=(-200, 200, -200, 200),
                 vmin=minmin, vmax=maxmax)
    ax[0].set_xlabel('X-coordinate')
    ax[0].set_ylabel('Y-coordinate')
    ax[0].set_title("Z=0m")
    ax[1].imshow(ps[1], origin="lower", extent=(-200, 200, -200, 200),
                 vmin=minmin, vmax=maxmax)
    ax[1].set_xlabel('X-coordinate')
    ax[1].set_ylabel('Y-coordinate')
    ax[1].set_title("Z=60m")
    plt.show()

    # Watch the centerline in time
    T = np.linspace(0, 24 * 3600, 10)
    for t in T:
        mtt.solve(t=t)
        plt.plot(mtt.x_disp(t=t),
                 mtt.z_disp(t=t),
                 label=str(round(curfunc(0, t), 2)) + ' m/s')
    plt.legend()
    plt.title("Centerline Over Time")
    plt.xlabel("Along Plume X-coordinate")
    plt.xlabel("Z-coordinate")
    plt.show()

    # Cache the environment, then read from it
    mtt.write_cache(tvec=[0, 75],
                    xrange=(0, 200),
                    yrange=(0, 200),
                    zrange=(0, 200),
                    xres=100,
                    yres=100,
                    zres=10)

    query_point = (np.linspace(10, 10, 1),
                   np.linspace(10, 10, 1),
                   np.linspace(200, 200, 1))

    print("Value from model: ", mtt.get_value(t=75,
                                              loc=query_point,
                                              from_cache=False))
    print("Value from cache: ", mtt.get_value(t=75,
                                              loc=query_point,
                                              from_cache=True,
                                              cache_interp="lookup"))

    Plive = mtt.get_snapshot(t=75,
                             z=[50],
                             xrange=(0, 200),
                             yrange=(0, 200),
                             xres=100,
                             yres=100,
                             from_cache=False)
    Pcache = mtt.get_snapshot(t=75,
                              z=[50],
                              xrange=(0, 200),
                              yrange=(0, 200),
                              xres=100,
                              yres=100,
                              from_cache=True)
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    minmin = np.nanmin(Plive.flatten())
    maxmax = np.nanmax(Plive.flatten())
    ax[0].imshow(Plive[0], origin="lower", extent=(0, 200, 0, 200),
                 vmin=minmin, vmax=maxmax)
    ax[0].set_title("In-Memory Snapshot")
    ax[1].imshow(Pcache[0], origin="lower", extent=(0, 200, 0, 200),
                 vmin=minmin, vmax=maxmax)
    ax[1].set_title("Cached Snapshot")
    plt.show()

###################################################
# Create Multiplume and plot environment outputs
###################################################
if RUN_MULTIPLUME is True:
    s = np.linspace(0, 1000, 100)
    extent = Extent(xrange=(0., 200.),
                    xres=100,
                    yrange=(0., 200.),
                    yres=100,
                    zrange=(0, 50),
                    zres=10,
                    global_origin=loc)
    mp1 = CrossflowMTT(plume_loc=(0, 0, 0),
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
    mp2 = CrossflowMTT(plume_loc=(75, 75, 0.),
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
    multiplume = Multiplume([mp1, mp2])

    # Get a plume intersection
    y = np.linspace(-200, 200, 1000)
    x = np.zeros_like(y)
    height = np.zeros(y.shape[0] * x.shape[0])
    pq = multiplume.get_value(t=10, loc=(x, y, height), from_cache=False)
    plt.plot(y, pq)
    plt.xlabel('Y-coordinate')
    plt.ylabel('Plume-State')
    plt.title('Environment Slice at X=0')
    plt.show()

    # Get a birds-eye snapshot of both plume probability
    ps = multiplume.get_snapshot(t=60, z=[10, 60])
    fig, ax = plt.subplots(3, 2, sharex=True, sharey=True)
    minmin = np.nanmin(ps.flatten())
    maxmax = np.nanmax(ps.flatten())
    ax[0, 0].imshow(ps[0], origin="lower", extent=(0, 200, 0, 200),
                    vmin=minmin, vmax=maxmax)
    ax[0, 0].set_ylabel('Y-coordinate')
    ax[0, 0].set_title("Z=0m, Both Plumes")
    ax[0, 1].imshow(ps[1], origin="lower", extent=(0, 200, 0, 200),
                    vmin=minmin, vmax=maxmax)
    ax[0, 1].set_ylabel('Y-coordinate')
    ax[0, 1].set_title("Z=40m, Both Plumes")

    # Get a birds-eye snapshot of one plume probability
    ps = multiplume.models[0].get_snapshot(t=60, z=[10, 60])
    ax[1, 0].imshow(ps[0], origin="lower", extent=(0, 200, 0, 200),
                    vmin=minmin, vmax=maxmax)
    ax[1, 0].set_ylabel('Y-coordinate')
    ax[1, 0].set_title("Z=0m, Plume A")
    ax[1, 1].imshow(ps[1], origin="lower", extent=(0, 200, 0, 200),
                    vmin=minmin, vmax=maxmax)
    ax[1, 1].set_ylabel('Y-coordinate')
    ax[1, 1].set_title("Z=40m, Plume A")

    # Get a birds-eye snapshot of one plume probability
    ps = multiplume.models[1].get_snapshot(t=60, z=[10, 40])
    ax[2, 0].imshow(ps[0], origin="lower", extent=(0, 200, 0, 200),
                    vmin=minmin, vmax=maxmax)
    ax[2, 0].set_xlabel('X-coordinate')
    ax[2, 0].set_ylabel('Y-coordinate')
    ax[2, 0].set_title("Z=0m, Plume B")
    ax[2, 1].imshow(ps[1], origin="lower", extent=(0, 200, 0, 200),
                    vmin=minmin, vmax=maxmax)
    ax[2, 1].set_xlabel('X-coordinate')
    ax[2, 1].set_ylabel('Y-coordinate')
    ax[2, 1].set_title("Z=40m, Plume B")
    plt.show()

    # Cache the environment, then read from it
    multiplume.write_cache(tvec=[0, 75],
                           xrange=(0, 200),
                           yrange=(0, 200),
                           zrange=(0, 200),
                           xres=100,
                           yres=100,
                           zres=10)

    query_point = (np.linspace(10., 10., 1),
                   np.linspace(10., 10., 1),
                   np.linspace(200., 200., 1))
    print("Value from model: ", multiplume.get_value(t=75,
                                                     loc=query_point,
                                                     from_cache=False))
    print("Value from cache: ", multiplume.get_value(
        t=75,
        loc=query_point,
        from_cache=True,
        cache_interp="lookup"))

    Plive = multiplume.get_snapshot(t=75,
                                    z=[50],
                                    xrange=(0, 200),
                                    yrange=(0, 200),
                                    xres=100,
                                    yres=100,
                                    from_cache=False)
    Pcache = multiplume.get_snapshot(t=75,
                                     z=[50],
                                     xrange=(0, 200),
                                     yrange=(0, 200),
                                     xres=100,
                                     yres=100,
                                     from_cache=True)
    f, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    minmin = np.nanmin(Plive.flatten())
    maxmax = np.nanmax(Plive.flatten())
    ax[0].imshow(Plive[0], origin="lower", extent=(0, 200, 0, 200),
                 vmin=minmin, vmax=maxmax)
    ax[0].set_title("In-Memory Snapshot")
    ax[1].imshow(Pcache[0], origin="lower", extent=(0, 200, 0, 200),
                 vmin=minmin, vmax=maxmax)
    ax[1].set_title("Cached Snapshot")
    plt.show()
