import netCDF4 as nc
from constants import weather_path, height_path, glob_fire_dir
from functools import lru_cache
from datetime import datetime, timedelta
import os


def calculate_restrictions():
    dsh = nc.Dataset(height_path)
    dsw = nc.Dataset(weather_path)

    height_restrictions = {
        "bottom": dsh['lat'][:].min(),
        "top": dsh['lat'][:].max(),
        "left": dsh['lon'][:].min(),
        "right": dsh['lon'][:].max(),
    }

    hours_start = dsw['time'][:].min().item()
    hours_end = dsw['time'][:].max().item()
    weather_abs_start = datetime(1900, 1, 1, 0, 0, 0, 0)
    start = weather_abs_start + timedelta(hours=hours_start)
    end = weather_abs_start + timedelta(hours=hours_end)

    weather_restrictions = {
        "bottom": dsw['latitude'][:].min(),
        "top": dsw['latitude'][:].max(),
        "left": dsw['longitude'][:].min(),
        "right": dsw['longitude'][:].max(),
        "time_start": start,
        "time_end": end,
    }

    # First data for glob fire 3 is january 2000
    start_year = 2000
    fire_paths = []
    months_since_start_year = 0
    while True:
        year = months_since_start_year // 12
        month = months_since_start_year % 12

        d = datetime(start_year + year, month + 1, 1)
        if weather_restrictions['time_start'] <= d <= weather_restrictions['time_end']:
            glob_fire_path = os.path.join(glob_fire_dir, f"MODIS_BA_GLOBAL_1_{d.month}_{d.year}.shp")
            if os.path.isfile(glob_fire_path):
                fire_paths.append(glob_fire_path)

        months_since_start_year += 1
        if months_since_start_year > (datetime.now().year - start_year + 1) * 12:
            break

    return height_restrictions, weather_restrictions, fire_paths


@lru_cache
def get_restrictions():
    height, weather, fire_paths = calculate_restrictions()
    return {
        "bottom": max(height['bottom'], weather['bottom']),
        "top": min(height['top'], weather['top']),
        "left": max(height['left'], weather['left']),
        "right": min(height['right'], weather['right']),
        "time_start": weather['time_start'],
        "time_end": weather['time_end'],
        "fire_paths": fire_paths,
    }
