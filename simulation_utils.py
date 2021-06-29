import os
import netCDF4 as nc
import geopandas
import numpy as np
from matplotlib import pyplot as plt
import csv
import matplotlib.patches as patches
import cv2 as cv
from math import sin, cos, sqrt, atan2, radians
from regions import region


def distance_between_coordinates(lat1, lon1, lat2, lon2):
    # approximate radius of earth in km
    earth_radius = 6373

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    lon_distance = lon2 - lon1
    lat_distance = lat2 - lat1

    a = sin(lat_distance / 2) ** 2 + cos(lat1) * cos(lat2) * sin(lon_distance / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return earth_radius * c


def load_burnt_area(nc_path, width, height, frame):
    [left, bottom, right, top] = frame.total_bounds

    ds = nc.Dataset(nc_path)
    layer = 'FDOB_DEKAD'
    # x = 71600
    # y = 15150  # lower is higher in the map
    # size = 400
    # bottom = y - size // 2
    # top = y + size // 2
    # left = x - size // 2
    # right = x + size // 2
    lats = ds['lat'][:].filled(0)
    lons = ds['lon'][:].filled(0)
    i_left = np.argmax(lons >= left)
    i_right = np.argmax(lons >= right)
    i_bottom = np.argmax(lats <= bottom)
    i_top = np.argmax(lats <= top)
    burns = ds[layer][i_top:i_bottom, i_left:i_right].filled(0)
    return cv.resize(burns, (height, width))


def load_burnt_areas(frame, width, height):
    nc_paths = region["nc_paths"]
    return load_burnt_area(nc_paths[0], width, height, frame), load_burnt_area(nc_paths[1], width, height, frame)


def load_land_cover_dataframe(frame):
    p = os.path.abspath("data/DATA/U2018_CLC2018_V2020_20u1.gpkg")
    print(p)
    # gdf_mask = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    # gdf_mask.plot()
    # gdf = geopandas.read_file(p, mask=gdf_mask[gdf_mask.name == "Cyprus"])
    gdf = geopandas.read_file(p, bbox=frame)
    gdf.to_crs(epsg=4326, inplace=True)

    # gdf = gdf.to_crs("EPSG:4326")

    [left, bottom, right, top] = frame.total_bounds
    rect = patches.Rectangle((left, bottom), (right - left), (top - bottom),
                             linewidth=1, edgecolor='r', facecolor='none')
    fig, ax = plt.subplots()
    gdf.plot("Code_18", ax=ax)
    ax.add_patch(rect)
    plt.show()
    print("Land cover data loaded")

    return gdf


def load_land_cover(frame, width, height, cached=True):
    land_cover_rates = {}
    with open('landCoverSpreadRate.csv') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for [clc_code, _, spread_rate] in reader:
            if clc_code == 'CLC_CODE':
                continue
            land_cover_rates[clc_code] = float(spread_rate)

    [left, bottom, right, top] = frame.total_bounds
    gdf = load_land_cover_dataframe(frame)

    lct_file = f"lct_{width}_{height}_{left}_{bottom}_{right}_{top}.npy"
    if not os.path.isfile(lct_file) or not cached:
        land_cover_types = np.zeros((width, height), np.int16)
        for y in range(0, height):
            for x in range(0, width):
                map_x = left + (right - left) * y / height
                map_y = bottom + (top - bottom) * (1 - x / width)
                map_value = gdf.cx[map_x:map_x + 0.0001, map_y:map_y + 0.0001]
                slice_length = len(map_value)
                if slice_length > 0:
                    cell_type = map_value.Code_18.values[0]
                else:
                    cell_type = 999
                land_cover_types[x, y] = cell_type
            print(f"{y + 1} / {height}")
        np.save(lct_file, land_cover_types)

    return land_cover_rates, np.load(lct_file)
