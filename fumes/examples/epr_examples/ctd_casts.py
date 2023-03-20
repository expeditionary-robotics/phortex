"""Process CTD casts"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fumes.environment.profile import Profile


def get_bottles(df, corrected_pos=None):
    """Helper to get bottle data cleaned out into it's own CSV file."""
    info = dict()
    for bot in np.unique(df.bottle_pos):
        if bot == 0:
            pass
        else:
            temp = df[df.bottle_pos == bot]
            usbl_info = corrected_pos[corrected_pos["bottle_pos"] == bot]
            data = temp.iloc[0]
            data["usbl_lat"] = usbl_info["lat"].values[0]
            data["usbl_lon"] = usbl_info["lon"].values[0]
            info[bot] = data
    bottle_df = pd.DataFrame.from_dict(info, orient="index")
    return bottle_df


# Get the profile names
FPATH = os.path.join(os.getenv("EPR_DATA"), "ctd/raw/")
# FNAMES = ["at5007007"]
# SKIPROWS = [285]
FNAMES = ["at5007001", "at5007002", "at5007004", "at5007007"]
SKIPROWS = [285, 278, 285, 285]
COL_NAMES = ["time",
             "bottle_pos",
             "nbf",
             "conductivity",
             "density",
             "fluor",
             "depth",
             "latitude",
             "longitude",
             "oxygenumol/kg",
             "pot_temp",
             "turbidity",
             "salinity",
             "flag"]
SKIP = 50


if __name__ == "__main__":
    casts = []
    for skip, fname in zip(SKIPROWS, FNAMES):
        # read in the table
        df = pd.read_table(os.path.join(FPATH, f"{fname}.cnv"), skiprows=skip, names=COL_NAMES, delim_whitespace=True)

        # create bottles df
        # corrected_locations = pd.read_table(os.path.join(FPATH, f"{fname}_points.txt"), names=["bottle_pos", "lat", "lon", "depth"], delimiter=",")
        # bottle_df = get_bottles(df, corrected_locations)
        # bottle_df.to_csv(os.path.join(os.getenv("EPR_DATA"), f"ctd/proc/{fname}_bottles.csv"))

        # Only grab the depth ranges that matter
        print(df)
        df = df[df.depth > 2000]

        # collect the cast information
        casts.append(df)


    # compare casts
    for i, cast in enumerate(casts):
        fig, ax = plt.subplots(1, 4, figsize=(10, 10), sharey=True)
        ax[0].plot(cast.pot_temp, cast.depth)
        ax[0].set_ylabel("Depth, m")
        ax[0].set_xlabel("Potential Temperature, C")
        ax[1].plot(cast.salinity, cast.depth)
        ax[1].set_xlabel("Practical Salinity, PSU")
        ax[2].plot(cast.turbidity, cast.depth)
        ax[2].set_xlabel("Turbidity, NTU")
        ax[3].plot(cast["oxygenumol/kg"], cast.depth)
        ax[3].set_xlabel("Oxygen, umol/kg")
        plt.ylim([2560, 2000])
        plt.title(FNAMES[i])
    plt.show()

    # create large frame of data and save
    casts_df = pd.concat(casts, axis=0, ignore_index=False)
    casts_df.to_csv(os.path.join(os.getenv("EPR_DATA"), "ctd/proc/cast_training_data.csv"))

    # check training quality
    datax = 2555. - casts_df.depth
    print(datax)
    datayt = casts_df.pot_temp
    datays = casts_df.salinity

    tprof = Profile(datax[::SKIP], datayt[::SKIP], training_iter=30, learning_rate=0.1)
    sprof = Profile(datax[::SKIP], datays[::SKIP], training_iter=30, learning_rate=0.1)

    plt.scatter(datayt, datax, color="orange", s=0.5, label="Data")
    plt.plot(tprof.profile(np.linspace(0, 1000, 100)), np.linspace(0, 1000, 100), label="Learned Function")
    plt.ylim([0, 1500])
    plt.xlabel("Potential Temperature, C")
    plt.ylabel("Depth, m")
    plt.legend()
    plt.show()

    plt.scatter(datays, datax, color="orange", s=0.5, label="Data")
    plt.plot(sprof.profile(np.linspace(0, 1000, 100)), np.linspace(0, 1000, 100), label="Learned Function")
    plt.ylim([0, 1500])
    plt.xlabel("Salinity, PSU")
    plt.ylabel("Depth, m")
    plt.legend()
    plt.show()
