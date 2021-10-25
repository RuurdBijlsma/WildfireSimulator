import netCDF4 as nc
import os
import numpy as np
from matplotlib import pyplot as plt


def load_weather(bounds):
    nc_path = os.path.abspath(
        "../data/Weather/adaptor.mars.internal-1630357570.1783972-1285-14-bb21d314-73d9-4199-a02a-c883a678b6de.nc")
    ds = nc.Dataset(nc_path)

    print(ds)
    lats = ds['latitude'][:].filled(0)
    lons = ds['longitude'][:].filled(0)
    i_left = np.argmax(lons >= bounds['left'])
    i_right = np.argmax(lons >= bounds['right'])
    i_bottom = np.argmax(lats <= bounds['bottom'])
    i_top = np.argmax(lats <= bounds['top'])

    weather = {}
    layers = ['u10', 'v10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'src', 'stl1', 'str', 'tp', 'swvl1']
    for layer in layers:
        weather[layer] = ds[layer][:, i_top:i_bottom, i_left:i_right].filled(0)

        # plt.imshow(weather[layer][0, :, :], interpolation='nearest')
        # title = os.path.basename(nc_path)
        # plt.title(title)
        # plt.show()

    return weather
