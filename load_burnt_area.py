import geopandas
import os
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import xarray as xr


def run():
    p = os.path.abspath("data/burnt/c_gls_BA300_201607200000_GLOBE_PROBAV_V1.0.1.nc")
    dnc = xr.open_dataset(p)
    print(f"lat size: {dnc.lat.size}")
    print(f"lon size: {dnc.lon.size}")
    d1d = dnc.isel(lat=10)
    d1d.plot.scatter(x=d1d.lon)

    # airtemps = xr.tutorial.open_dataset("air_temperature")
    # air = airtemps.air - 273.15
    #
    # air1d = air.isel(lat=10, lon=10)
    # air1d.plot()
    #
    # air2d = air.isel(time=500)
    # air2d.plot()
    # print(5)

    plt.show()

if __name__ == '__main__':
    run()
