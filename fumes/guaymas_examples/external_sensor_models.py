"""Computes current and background profile models used during field operations."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes
from fumes.environment.profile import Profile
from fumes.environment.current import CurrMag, CurrHead

DATA_HOME = "/home/vpreston/rrg/src/planning/sentry/data/"
TEMPERATURE_BKGND_FILENAME = os.path.join(DATA_HOME, "profiles/proc_temp_profile.csv")
SALINITY_BKGND_FILENAME = os.path.join(DATA_HOME, "profiles/proc_salt_profile.csv")
CURRENT_FILENAME = os.path.join(DATA_HOME, "currents/proc_current_train_profile1.csv")
REFERENCE = (float(os.getenv("LAT")),
             float(os.getenv("LON")),
             float(os.getenv("DEP")))
SR = 30  # subsample rate

temp = pd.read_csv(TEMPERATURE_BKGND_FILENAME)
datax_temp, datay_temp = temp['depth'], temp['temperature']
# datax_temp = -datax_temp + REFERENCE[2]
print("Training temperature...")
TPROF = Profile(datax_temp[::SR], datay_temp[::SR], training_iter=100, learning_rate=0.1)
# temperature plot
depth_reference = np.linspace(np.nanmin(datax_temp), np.nanmax(datax_temp), 1000)
fig, ax = plt.subplots(1, 2, sharey=True)
ax[0].scatter(datay_temp[::SR], datax_temp[::SR], alpha=0.05, color="gray")
ax[0].plot(TPROF.profile(depth_reference), depth_reference, color="red")
ax[0].set_xlabel("Temperature (C)")
ax[0].set_ylabel("Depth (m)")

salt = pd.read_csv(SALINITY_BKGND_FILENAME)
datax_salt, datay_salt = salt['depth'], salt['salinity']
# datax_salt = -datax_salt + REFERENCE[2]
print("Training salinity...")
SPROF = Profile(datax_salt[::SR], datay_salt[::SR], training_iter=100, learning_rate=0.1)
# salinity plot
ax[1].scatter(datay_salt[::SR], datax_salt[::SR], alpha=0.05, color="gray")
ax[1].plot(SPROF.profile(depth_reference), depth_reference, color="red")
ax[1].set_xlabel("Salinity (PSU)")
plt.show()

# current = pd.read_csv(CURRENT_FILENAME)
# current = current.dropna()
# datax_cur, datay_mag, datay_head = current['hours'], current['mag_mps_train'], current['head_rad_train']
# print("Training current magnitude...")
# CURRMAG = CurrMag(datax_cur, datay_mag, training_iter=100, learning_rate=0.5)
# # current mag plot
# time_reference = np.linspace(0, 23.9*3600, 50)
# fig, ax = plt.subplots(2, 1, sharex=True)
# ax[0].scatter(datax_cur, datay_mag, color="gray", alpha=0.1)
# ax[0].plot(time_reference / 3600. % 24, CURRMAG.magnitude(None, time_reference), color="red")
# ax[0].set_ylabel("Magnitude (m/s)")

# print("Training current heading...")
# CURRHEAD = CurrHead(datax_cur, datay_head * 180 / np.pi, training_iter=200, learning_rate=0.1)
# # current heading plot
# ax[1].scatter(datax_cur, (datay_head * 180 / np.pi) % 360, color="gray", alpha=0.1)
# ax[1].plot(time_reference / 3600. % 24, (CURRHEAD.heading(time_reference) * 180. / np.pi) % 360, color="red")
# ax[1].set_ylabel("Heading (deg)")
# ax[1].set_xlabel("Time (hr)")
# plt.show()
