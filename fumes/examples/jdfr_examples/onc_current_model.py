"""Model the currents in the JdFR"""

import os
import pandas as pd
import numpy as np
from fumes.environment import current
import datetime as datetime
import matplotlib.pyplot as plt

DIVE_NAME = "sentry685"


if __name__ == "__main__":
    # Read in current data
    cdata = pd.read_csv(os.path.join(os.getenv("JDFR_OUTPUT"), "sentry/ocn_data_prediction.csv"))
    cdata.Datetime = pd.to_datetime(cdata.Datetime)
    
    # Create test and train data
    datetime_start = datetime.datetime(year=2023, month=9, day=3, hour=0, minute=0, second=0, tzinfo=datetime.timezone.utc)
    datetime_ref = datetime.datetime(year=2023, month=9, day=5, hour=10, minute=0, second=0, tzinfo=datetime.timezone.utc)
    datetime_horizon = datetime.datetime(year=2023, month=9, day=6, hour=23, minute=0, second=0, tzinfo=datetime.timezone.utc)
    train_data = cdata[(cdata.Datetime > datetime_start) & (cdata.Datetime < datetime_ref)]
    test_data = cdata[cdata.Datetime >= datetime_ref]
    query_horizon = (pd.to_datetime(datetime_horizon) - pd.Timestamp("1970-01-01", tzinfo=datetime.timezone.utc)) // pd.Timedelta("1s")
    query_points = np.linspace(cdata.t.values[-1], query_horizon, 24*60*60)
    forecast_dates = [pd.to_datetime(d, unit="s") for d in query_points]

    # train for Vnorth and Veast
    print("Training East...")
    cur_east = current.CurrMag((train_data.t / 3600.) % 24, train_data.VEast, training_iter=10, learning_rate=0.1, type="Periodic")
    print("Training North...")
    cur_north = current.CurrMag((train_data.t / 3600.) % 24, train_data.VNorth, training_iter=10, learning_rate=0.1, type="Periodic")

    # validate
    print("Generating Predictions...")
    pred_east = cur_east.magnitude(None, (cdata.t.values / 3600.) % 24)
    pred_north = cur_north.magnitude(None, (cdata.t.values / 3600.) % 24)
    forecast_east = cur_east.magnitude(None, (query_points / 3600.) % 24)
    forecast_north = cur_north.magnitude(None, (query_points / 3600.) % 24)

    # plot
    print("Plotting...")
    plt.plot(cdata.Datetime, cdata.VEast, label="Real")
    plt.plot(cdata.Datetime, pred_east, label="Model Predicted")
    plt.plot(forecast_dates, forecast_east, label="Forecast")
    plt.show()

    plt.plot(cdata.Datetime, cdata.VNorth, label="Real")
    plt.plot(cdata.Datetime, pred_north, label="Model Predicted")
    plt.plot(forecast_dates, forecast_north, label="Forecast")
    plt.show()

    plt.plot(cdata.Datetime, np.sqrt(cdata.VEast ** 2 + cdata.VNorth **2), label="Real")
    plt.plot(cdata.Datetime, np.sqrt(pred_east ** 2 + pred_north **2), label="Predicted")
    plt.plot(forecast_dates, np.sqrt(forecast_east ** 2 + forecast_north **2), label="Forecast")
    plt.vlines(x=[datetime_ref], ymin=[0], ymax=[0.2])
    plt.show()

    