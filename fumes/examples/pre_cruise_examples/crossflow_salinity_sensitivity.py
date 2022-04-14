"""Instantiates all of the MTT environments in example."""

from re import X
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from fumes.environment.mtt import CrossflowMTT
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
# s0 = 34.608  # source salinity
s0 = np.linspace(34, 68, 35)
t0 = 300.  # source temperature
# t0 = np.linspace(100, 500, 20)
rho0 = eos_rho(t0, s0)  # source density

lam = 1.0  # for crossflow, major-minor axis ratio
entrainment = [0.12, 0.2]  # entrainment coeffs

s = np.linspace(0, 1000, 100)
extent = Extent(xrange=(-200., 200.),
                xres=100,
                yrange=(-200., 200.),
                yres=100,
                zrange=(0, 50),
                zres=10,
                global_origin=loc)

###################################################
# Create CrossflowMTT and plot environment outputs
###################################################
label = []
for s0meas in s0:
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
                       salt=s0meas,
                       temp=t0,
                       density=rho0,
                       lam=lam,
                       entrainment=entrainment)
    mtt.solve(t=6*3600)
    # print(eos_rho(t0s, s0))
    # Draw a vertical slice of the plume envelope
    le, cl, re = mtt.envelope(t=6*3600)
    # plt.fill_betweenx(mtt.z_disp, le, re, alpha=0.5, label='Initialized Model')
    verts = [*zip(le[0], le[1]), *zip(reversed(re[0]), reversed(re[1]))]
    poly = Polygon(verts, alpha=0.1)
    plt.gca().add_patch(poly)
    label.append(str(s0meas))
    # print(s0meas)
plt.xlim([-100, 100])
plt.ylim([0, 500])
plt.legend(label)
plt.show()
