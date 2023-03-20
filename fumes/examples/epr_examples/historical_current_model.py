"""Create a current model for the YBW-Sentry, that at least mimics cadence and amplitudes."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

from fumes.environment.current import CurrMag, CurrHead


def process_historical_data_rr2102():
    # Read in the data for the latest cruise, which captured two january months of data for reference
    tiltmeter_rr2102_stadium = pd.read_table(os.path.join(os.getenv("EPR_DATA"), "current/tiltmeter_rr2102_stadium.csv"),
                                             names=["Datetime", "Speed",
                                                    "Direction", "Speed_N", "Speed_E"],
                                             delimiter=",",
                                             skiprows=1)
    tiltmeter_rr2102_stadium.loc[:, "Timestamp"] = pd.to_datetime(
        tiltmeter_rr2102_stadium.Datetime, utc=True, format="%Y-%m-%dT%H:%M:%S.%f")
    tiltmeter_rr2102_stadium.loc[:, "posixtime"] = tiltmeter_rr2102_stadium.apply(
        lambda x: x["Timestamp"].timestamp(), axis=1)
    tiltmeter_rr2102_stadium.loc[:, "Month"] = tiltmeter_rr2102_stadium.apply(
        lambda x: x["Timestamp"].month, axis=1)
    tiltmeter_rr2102_stadium.loc[:, "Day"] = tiltmeter_rr2102_stadium.apply(
        lambda x: x["Timestamp"].day, axis=1)
    tiltmeter_rr2102_stadium.loc[:, "Hour"] = tiltmeter_rr2102_stadium.apply(
        lambda x: x["Timestamp"].hour, axis=1)
    tiltmeter_rr2102_stadium.loc[:, "Year"] = tiltmeter_rr2102_stadium.apply(
        lambda x: x["Timestamp"].year, axis=1)

    # Separate out the two months
    rr21_20 = tiltmeter_rr2102_stadium[(tiltmeter_rr2102_stadium.Year == 2020) & (
        tiltmeter_rr2102_stadium.Month == 1)].reset_index()
    rr21_21 = tiltmeter_rr2102_stadium[(tiltmeter_rr2102_stadium.Year == 2021) & (
        tiltmeter_rr2102_stadium.Month == 1)].reset_index()

    # Create helper hours bin
    rr21_20.loc[:, "Cumulative_Hours"] = rr21_20.apply(lambda x: (
        x["posixtime"] - np.nanmin(rr21_20["posixtime"])) / 3600., axis=1)
    rr21_21.loc[:, "Cumulative_Hours"] = rr21_21.apply(lambda x: (
        x["posixtime"] - np.nanmin(rr21_21["posixtime"])) / 3600., axis=1)

    rr21_20.to_csv(os.path.join(os.getenv("EPR_DATA"),
                   "current/tiltmeter_rr2102_stadium_processed_202001.csv"))
    rr21_21.to_csv(os.path.join(os.getenv("EPR_DATA"),
                   "current/tiltmeter_rr2102_stadium_processed_202101.csv"))

    return rr21_20, rr21_21


def process_historical_data(fname, labels):
    # read in the data
    print("Reading data...")
    data_df = pd.read_table(os.path.join(os.getenv("EPR_DATA"), f"current/{fname}.csv"),
                            names=labels,
                            delimiter=",",
                            skiprows=1)

    # add timing info
    print("Adding timestamp...")
    data_df.loc[:, "Timestamp"] = pd.to_datetime(
        data_df.Datetime, utc=True, format="%Y-%m-%dT%H:%M:%S.%f")
    print("Adding posixtime...")
    data_df.loc[:, "posixtime"] = data_df.apply(lambda x: x["Timestamp"].timestamp(), axis=1)

    # save for access later
    print("Saving...")
    data_df.to_csv(os.path.join(os.getenv("EPR_DATA"), f"current/{fname}_processed.csv"))
    return data_df


def read_historical_data(fname):
    cruise_df = pd.read_csv(os.path.join(os.getenv("EPR_DATA"), f"current/{fname}_processed.csv"))
    cruise_df.loc[:, "Timestamp"] = pd.to_datetime(cruise_df["Timestamp"])
    return cruise_df


if __name__ == "__main__":
    # Params
    fname = "2012823_TCM3-sn2012823_(0)_Current_declination6degE"
    timestart = datetime(2023, 1, 14, 0, 0, 2, tzinfo=timezone.utc)
    timeend = datetime(2023, 1, 18, 2, 30, 2, tzinfo=timezone.utc)
    pN = 1  # how many items to skip for plotting purposes

    # Process historical data
    # current_df = process_historical_data(fname=fname,
    #                                      labels=["Datetime", "Speed", "Heading", "NVel", "Evel"])
    
    # Read in processed historical data
    current_df = read_historical_data(f"{fname}")


    # Grab only the time period you care about
    current_df.set_index("Timestamp", inplace=True)
    current_df = current_df.loc[timestart:timeend]

    # Scrape out suspicious data
    current_df = current_df[current_df.Speed > 5.]


    # Create helper field for training
    min_posixt = np.nanmin(current_df.posixtime)
    current_df.loc[:, "Cumulative_Hours"] = current_df.apply(lambda x: (x.posixtime - min_posixt) / 3600., axis=1)

    # Plot a direction and magnitude plot
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].scatter(current_df.index[::pN], current_df.Heading[::pN])
    ax[0].set_title("Heading over Time")
    ax[0].set_ylabel("Heading, Degrees")
    ax[1].scatter(current_df.index[::pN], current_df.Speed[::pN])
    ax[1].set_title("Magnitude over Time")
    ax[1].set_ylabel("Speed, cm/s")
    ax[1].set_xlabel("Time")
    plt.show()

    # print(rr21_20.Timestamp[0:10])
    # print(rr21_21.Timestamp[0:10])
    # plt.plot(rr21_20.Cumulative_Hours, rr21_20.Speed)
    # plt.plot(rr21_21.Cumulative_Hours, rr21_21.Speed)
    # # plt.plot(rr21_21.Speed)
    # plt.show()

    # Create GP model for Speed
    # TRAIN_N = 1
    # WINDOW = 2 * 24 * 60
    # cmag = CurrMag(rr21_20.Cumulative_Hours.values[:WINDOW:TRAIN_N],
    #                rr21_20.Speed.values[:WINDOW:TRAIN_N],
    #                training_iter=200,
    #                learning_rate=0.1,
    #                type="Periodic")
    # plt.scatter(rr21_20.Cumulative_Hours[:WINDOW:TRAIN_N],
    #             rr21_20.Speed[:WINDOW:TRAIN_N], c='orange')
    # plt.plot(rr21_20.Cumulative_Hours[:WINDOW], cmag.magnitude(
    #     None, rr21_20.posixtime[:WINDOW] - np.nanmin(rr21_20.posixtime)))
    # plt.show()

    # # Create GP model for Direction (need to unwrap and then wrap)
    # chead = CurrHead(rr21_20.Cumulative_Hours.values[:WINDOW:TRAIN_N],
    #                  180 / np.pi * np.unwrap(np.radians(rr21_20.Direction.values[:WINDOW:TRAIN_N])),
    #                  training_iter=100,
    #                  learning_rate=0.1,
    #                  type="Periodic")
    # plt.scatter(rr21_20.Cumulative_Hours[:WINDOW:TRAIN_N], 180 / np.pi *
    #             np.unwrap(np.radians(rr21_20.Direction.values[:WINDOW:TRAIN_N])), c="orange")
    # plt.plot(rr21_20.Cumulative_Hours[:WINDOW], (180. / np.pi *
    #          chead.heading(rr21_20.posixtime[:WINDOW] - np.nanmin(rr21_20.posixtime))))
    # plt.show()
