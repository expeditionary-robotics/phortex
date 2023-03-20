"""Analyzes historical current data from the EPR."""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get the data files
adcp_9495 = pd.read_table(os.path.join(os.getenv("EPR_DATA"), "current/adcp_1994_1995.txt"),
                          delimiter=" ",
                          names=["Date", "Time", "Temp", "Direction",
                                 "Speed", "Speed_N", "Speed_E"],
                          skiprows=1)
adcp_9495.loc[:, "Timestamp"] = adcp_9495.apply(lambda x: pd.to_datetime(f"{x.Date}T{x.Time}", utc=True, format="%Y/%m/%dT%H:%M:%S"), axis=1)
adcp_9495.loc[:, "Month"]  = adcp_9495.apply(lambda x: x["Timestamp"].month, axis=1)
adcp_9495.loc[:, "Day"]  = adcp_9495.apply(lambda x: x["Timestamp"].day, axis=1)
adcp_9495.loc[:, "Hour"]  = adcp_9495.apply(lambda x: x["Timestamp"].hour, axis=1)

adcp_9899 = pd.read_table(os.path.join(os.getenv("EPR_DATA"), "current/adcp_1998_1999.txt"),
                          delimiter=" ",
                          names=["Date", "Time", "Temp", "Direction",
                                 "Speed", "Speed_N", "Speed_E"],
                          skiprows=1)
adcp_9899.loc[:,"Timestamp"] = adcp_9899.apply(lambda x: pd.to_datetime(f"{x.Date}T{x.Time}", utc=True, format="%Y/%m/%dT%H:%M:%S"), axis=1)
adcp_9899.loc[:, "Month"]  = adcp_9899.apply(lambda x: x["Timestamp"].month, axis=1)
adcp_9899.loc[:, "Day"]  = adcp_9899.apply(lambda x: x["Timestamp"].day, axis=1)
adcp_9899.loc[:, "Hour"]  = adcp_9899.apply(lambda x: x["Timestamp"].hour, axis=1)

adcp_2000 = pd.read_table(os.path.join(os.getenv("EPR_DATA"), "current/adcp_2000.txt"),
                          delimiter=" ",
                          names=["Date", "Time", "Temp", "Direction",
                                 "Speed", "Speed_N", "Speed_E"],
                          skiprows=1)
adcp_2000.loc[:,"Timestamp"] = adcp_2000.apply(lambda x: pd.to_datetime(f"{x.Date}T{x.Time}", utc=True, format="%Y/%m/%dT%H:%M:%S"), axis=1)
adcp_2000.loc[:, "Month"]  = adcp_2000.apply(lambda x: x["Timestamp"].month, axis=1)
adcp_2000.loc[:, "Day"]  = adcp_2000.apply(lambda x: x["Timestamp"].day, axis=1)
adcp_2000.loc[:, "Hour"]  = adcp_2000.apply(lambda x: x["Timestamp"].hour, axis=1)

tiltmeter_ad4206 = pd.read_table(os.path.join(os.getenv("EPR_DATA"), "current/tiltmeter_at4206.txt"),
                                 names=["Datetime", "Speed", "Direction", "Speed_N", "Speed_E"],
                                 delimiter=",",
                                 skiprows=1)
tiltmeter_ad4206.loc[:,"Timestamp"] = pd.to_datetime(tiltmeter_ad4206.Datetime, utc=True, format="%Y-%m-%dT%H:%M:%S.%f")
tiltmeter_ad4206.loc[:, "Month"]  = tiltmeter_ad4206.apply(lambda x: x["Timestamp"].month, axis=1)
tiltmeter_ad4206.loc[:, "Day"]  = tiltmeter_ad4206.apply(lambda x: x["Timestamp"].day, axis=1)
tiltmeter_ad4206.loc[:, "Hour"]  = tiltmeter_ad4206.apply(lambda x: x["Timestamp"].hour, axis=1)

tiltmeter_rr2102_stadium = pd.read_table(os.path.join(os.getenv("EPR_DATA"), "current/tiltmeter_rr2102_stadium.csv"),
                                         names=["Datetime", "Speed",
                                                "Direction", "Speed_N", "Speed_E"],
                                         delimiter=",",
                                         skiprows=1)
tiltmeter_rr2102_stadium.loc[:,"Timestamp"] = pd.to_datetime(tiltmeter_rr2102_stadium.Datetime, utc=True, format="%Y-%m-%dT%H:%M:%S.%f")
tiltmeter_rr2102_stadium.loc[:, "Month"]  = tiltmeter_rr2102_stadium.apply(lambda x: x["Timestamp"].month, axis=1)
tiltmeter_rr2102_stadium.loc[:, "Day"]  = tiltmeter_rr2102_stadium.apply(lambda x: x["Timestamp"].day, axis=1)
tiltmeter_rr2102_stadium.loc[:, "Hour"]  = tiltmeter_rr2102_stadium.apply(lambda x: x["Timestamp"].hour, axis=1)


# tiltmeter_rr2102_vvent = pd.read_table(os.path.join(os.getenv("EPR_DATA"), "current/tiltmeter_rr2102_vvent.csv"),
#                                        names=["Datetime", "Speed",
#                                               "Direction", "Speed_N", "Speed_E"],
#                                        delimiter=",",
#                                        skiprows=1)
# tiltmeter_rr2102_vvent.loc[:,"Timestamp"] = pd.to_datetime(tiltmeter_rr2102_vvent.Datetime, utc=True, format="%Y-%m-%dT%H:%M:%S.%f")
# tiltmeter_rr2102_vvent.loc[:, "Month"]  = tiltmeter_rr2102_vvent.apply(lambda x: x["Timestamp"].month, axis=1)
# tiltmeter_rr2102_vvent.loc[:, "Day"]  = tiltmeter_rr2102_vvent.apply(lambda x: x["Timestamp"].day, axis=1)
# tiltmeter_rr2102_vvent.loc[:, "Hour"]  = tiltmeter_rr2102_vvent.apply(lambda x: x["Timestamp"].hour, axis=1)

current_data = [adcp_9495, adcp_9899, adcp_2000, tiltmeter_ad4206, tiltmeter_rr2102_stadium]#, tiltmeter_rr2102_vvent]
# for cd in current_data:
#     plt.plot(cd.Timestamp, cd.Speed)
# plt.show()

# for cd in current_data:
#     plt.plot(cd.Timestamp, cd.Direction)
# plt.show()

for cd in current_data:
    print(cd["Timestamp"])
    temp = cd[cd.Month == 1]
    # plt.scatter(temp.Day, temp.Direction)
    plt.scatter(temp.Direction, temp.Speed)
plt.legend(["9495", "9899", "2000", "ad4206", "rr2102stad"])#, "rr2102vvent"])
plt.show()

plt.scatter(temp.Timestamp, np.unwrap(temp.Direction))
plt.show()