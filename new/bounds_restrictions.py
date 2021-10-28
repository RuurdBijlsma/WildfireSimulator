import netCDF4 as nc
from data_paths import weather_path, height_path
from functools import lru_cache


def calculate_restrictions():
    dsh = nc.Dataset(height_path)
    dsw = nc.Dataset(weather_path)

    height_restrictions = {
        "bottom": dsh['lat'][:].min(),
        "top": dsh['lat'][:].max(),
        "left": dsh['lon'][:].min(),
        "right": dsh['lon'][:].max(),
    }

    weather_restrictions = {
        "bottom": dsw['latitude'][:].min(),
        "top": dsw['latitude'][:].max(),
        "left": dsw['longitude'][:].min(),
        "right": dsw['longitude'][:].max(),
    }

    return height_restrictions, weather_restrictions


@lru_cache
def get_restrictions():
    height, weather = calculate_restrictions()
    return {
        "bottom": max(height['bottom'], weather['bottom']),
        "top": min(height['top'], weather['top']),
        "left": max(height['left'], weather['left']),
        "right": min(height['right'], weather['right']),
    }