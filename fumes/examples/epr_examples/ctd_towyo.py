"""Takes in Tow-YO CTD cast information and processes the plots."""

import os
import utm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from fumes.environment.utils import get_bathy


def get_bottles(df, corrected_pos=None):
    """Helper to get bottle data cleaned out into it's own CSV file."""
    info = dict()
    for bot in np.unique(df.bottle_pos):
        if bot == 0:
            pass
        else:
            temp = df[df.bottle_pos == bot]
            bottle_time = temp.Timestamp.iloc[0]
            corrected_pos.loc[:, "timediff"] = corrected_pos.apply(lambda x: x.Timestamp - bottle_time, axis=1)
            cor_pos = corrected_pos[corrected_pos.Timestamp >= bottle_time]
            min_diff = np.nanmin(cor_pos.timediff)
            usbl_info = corrected_pos[corrected_pos.timediff == min_diff]
            data = temp.iloc[0]
            data["usbl_lat"] = usbl_info["usbl_lat"].values[0]
            data["usbl_lon"] = usbl_info["usbl_lon"].values[0]
            info[bot] = data
    bottle_df = pd.DataFrame.from_dict(info, orient="index")
    return bottle_df


def process_usbl_udp(path, fnames):
    """Helper to strip out the useful CTD location information from logged USBL messages."""
    info = dict()
    j = 0
    for fname in fnames:
        with open(os.path.join(path, fname), "r") as uf:
            for line in uf:
                if "Elevator2" in line:
                    splits = line.split(" ")
                    info[j] = dict(datetime=f"{splits[1]}T{splits[2]}",
                                   usbl_lon=splits[5],
                                   usbl_lat=splits[6])
                    j += 1
    df = pd.DataFrame.from_dict(info, orient="index")
    df.loc[:, "Timestamp"] = pd.to_datetime(df.datetime, format="%Y/%m/%dT%H:%M:%S.%f", utc=True)
    return df


# Get the profile names
FPATH = os.path.join(os.getenv("EPR_DATA"), "ctd/raw/")
# SNAME = "towyo_003"
# CAST_FNAMES = ["at5007003"]
# CAST_START_TIME = [pd.to_datetime("2023-01-21T03:54:36.0", format="%Y-%m-%dT%H:%M:%S.%f", utc=True)]
# USBL_FNAMES = ["usbl_elevator2_ctd003.txt", "usbl_elevator2_ctd003_leg2.txt"]
# SNAME = "towyo_005"
# CAST_FNAMES = ["at5007005"]
# CAST_START_TIME = [pd.to_datetime("2023-01-24T04:35:38.0", format="%Y-%m-%dT%H:%M:%S.%f", utc=True)]
# USBL_FNAMES = ["towyo_elevator2_012323.txt"]
SNAME = "towyo_006"
CAST_FNAMES = ["at5007006"]
USBL_FNAMES = ["towyo3_elevator2_ctd006.txt"]
CAST_START_TIME = [pd.to_datetime("2023-01-25T19:59:02.0", format="%Y-%m-%dT%H:%M:%S.%f", utc=True)]
SKIPROWS = [285]
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

if __name__ == "__main__":
    # read in the cast data and adjust timestamp from utc time creation in cnv
    # casts = []
    # for skip, fname in zip(SKIPROWS, CAST_FNAMES):
    #     # read in the table
    #     df = pd.read_table(os.path.join(FPATH, f"{fname}.cnv"),
    #                        skiprows=skip, names=COL_NAMES, delim_whitespace=True)

    #     # Only grab the depth ranges that matter
    #     df = df[df.depth > 1500]

    #     # collect the cast information
    #     casts.append(df)

    # # create large frame of data and save
    # casts_df = pd.concat(casts, axis=0, ignore_index=False)
    # casts_df.loc[:, "Timestamp"] = casts_df.apply(
    #     lambda x: pd.to_timedelta(x.time, "S") + CAST_START_TIME[0], axis=1)
    # casts_df.to_csv(os.path.join(os.getenv("EPR_DATA"), f"ctd/proc/{SNAME}.csv"))
    # # print(casts_df)

    # # read in the USBL data and adjust timestamp
    # usbl_df = process_usbl_udp(FPATH, USBL_FNAMES)

    # # create bottles df
    # bottle_df = get_bottles(casts_df, usbl_df)
    # bottle_df.to_csv(os.path.join(os.getenv("EPR_DATA"), f"ctd/proc/{fname}_bottles.csv"))

    # # merge the data files together
    # casts_df.set_index("Timestamp", inplace=True)
    # usbl_df.set_index("Timestamp", inplace=True)
    # casts_df = casts_df.merge(usbl_df, how="outer", right_index=True, left_index=True)
    # print(casts_df)
    # casts_df["usbl_lat"] = casts_df["usbl_lat"].astype(np.float64)
    # casts_df["usbl_lon"] = casts_df["usbl_lon"].astype(np.float64)
    # casts_df = casts_df.interpolate(method="time")
    # casts_df.dropna(inplace=True)
    # print(casts_df)

    # # save merged file
    # casts_df.to_csv(os.path.join(os.getenv("EPR_DATA"), f"ctd/proc/{SNAME}_alldata.csv"))
    casts_df = pd.read_csv(os.path.join(os.getenv("EPR_DATA"), f"ctd/proc/{SNAME}_alldata.csv"))

    # plot
    plt.scatter(casts_df["fluor"], casts_df["depth"])
    plt.show()

    lower_column = casts_df[casts_df.depth > 2000]
    plt.hist(lower_column["fluor"], bins=100)
    plt.show()

    fig, ax = plt.subplots(1, 4, figsize=(10, 10), sharey=True)
    ax[0].scatter(lower_column.pot_temp, lower_column.depth, s=0.8)
    ax[0].set_ylabel("Depth, m")
    ax[0].set_xlabel("Potential Temperature, C")
    ax[1].scatter(lower_column.salinity, lower_column.depth, s=0.8)
    ax[1].set_xlabel("Practical Salinity, PSU")
    ax[2].scatter(lower_column["turbidity"], lower_column.depth, s=0.8)
    ax[2].set_xlabel("Turbidity, NTU")
    ax[3].scatter(lower_column["oxygenumol/kg"], lower_column.depth, s=0.8)
    ax[3].set_xlabel("Oxygen, umol/kg")
    plt.ylim([2600, 2000])
    plt.show()

    plt.scatter(lower_column.time, lower_column.depth, c=lower_column.turbidity, cmap="inferno")
    plt.show()

    # Globals
    BATHY = get_bathy(lat_min=np.nanmin(casts_df.usbl_lat),
                      lat_max=np.nanmax(casts_df.usbl_lat),
                      lon_min=np.nanmin(casts_df.usbl_lon),
                      lon_max=np.nanmax(casts_df.usbl_lon),
                      rsamp=0.1, buffer=0.001)
    Beast, Bnorth, _, _ = utm.from_latlon(BATHY.lat.values, BATHY.lon.values)
    bathy_plot = go.Mesh3d(x=Beast, y=Bnorth, z=BATHY.depth,
                           intensity=BATHY.depth,
                           colorscale='Viridis',
                           opacity=1.0,
                           name="Bathy")
    name = 'eye = (x:0., y:0., z:2.5)'
    camera = dict(eye=dict(x=1.25, y=1.25, z=0.1),
                  center=dict(x=0, y=0, z=-0.2))

    # tow fig
    TX, TY, _, _ = utm.from_latlon(lower_column.usbl_lat.values, lower_column.usbl_lon.values)
    tow_fig = go.Scatter3d(x=TX,
                           y=TY,
                           z=-lower_column.depth,
                           mode="markers",
                           marker=dict(size=0.8,
                                       color=lower_column.turbidity,
                                       colorscale="inferno",
                                       colorbar=dict(thickness=20, x=-0.3),
                                       cmin=np.percentile(lower_column.turbidity, 10),
                                       cmax=np.percentile(lower_column.turbidity, 90)))

    # Plot the 3D view of everything
    fig = go.Figure(data=[bathy_plot, tow_fig])  # , layout=layout)
    fig.update_layout(showlegend=True,
                      xaxis_title="Longitude",
                      yaxis_title="Latitude",
                      font=dict(size=18),
                      scene=dict(zaxis_title="",
                                 xaxis_title="",
                                 yaxis_title="",
                                 zaxis=dict(range=[-2580, -2520], tickfont=dict(size=20)),
                                 yaxis=dict(tickfont=dict(size=20)),
                                 xaxis=dict(tickfont=dict(size=20))),
                      scene_camera=camera)
    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.show()
