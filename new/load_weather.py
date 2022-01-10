import netCDF4 as nc
import os
import numpy as np
from matplotlib import pyplot as plt
from constants import weather_path


def load_weather(bounds):
    ds = nc.Dataset(weather_path)

    # print(ds)

    lats = ds['latitude'][:].filled(0)
    lons = ds['longitude'][:].filled(0)
    i_left = np.argmax(lons >= bounds['left'])
    i_right = np.argmax(lons >= bounds['right'])
    i_bottom = np.argmax(lats <= bounds['bottom'])
    i_top = np.argmax(lats <= bounds['top'])
    if i_left == i_right:
        i_right += 1
    if i_bottom == i_top:
        i_bottom += 1

    weather_dict = {}
    layers = ['u10', 'v10', 'd2m', 't2m', 'lai_hv', 'lai_lv', 'stl1', 'str', 'tp', 'swvl1']
    for layer in layers:
        weather_dict[layer] = ds[layer][:, i_top:i_bottom, i_left:i_right].filled(0)

        # plt.imshow(weather[layer][0, :, :], interpolation='nearest')
        # title = os.path.basename(nc_path)
        # plt.title(title)
        # plt.show()

    return weather_dict
