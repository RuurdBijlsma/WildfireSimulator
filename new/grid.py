import math
from datetime import timedelta
import numpy as np
from shapely.geometry import Point
import cv2
from datetime import datetime
import csv
import os
from matplotlib import pyplot as plt


class Grid:
    def __init__(self, bounds):
        # 1 grid cell is how big?: (coordinates)
        self.spatial_resolution = .001
        # 1 grid cell is how long?: (hours)
        self.temporal_resolution = timedelta(hours=1)

        self.coord_bounds = bounds
        self.coord_width = bounds['right'] - bounds['left']
        self.coord_height = bounds['right'] - bounds['left']
        self.timedelta = bounds['time_end'] - bounds['time_start']
        self.width = math.ceil(self.coord_width / self.spatial_resolution)
        self.height = math.ceil(self.coord_height / self.spatial_resolution)
        self.duration = math.ceil(self.timedelta / self.temporal_resolution)

    def land_cover_grid(self, land_cover_data, use_cache=True):
        land_cover_rates = {}
        with open('../landCoverSpreadRate.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for [clc_code, _, spread_rate] in reader:
                if clc_code == 'CLC_CODE':
                    continue
                land_cover_rates[clc_code] = float(spread_rate)

        [left, bottom, right, top] = [self.coord_bounds['left'], self.coord_bounds['bottom'],
                                      self.coord_bounds['right'], self.coord_bounds['top']]

        lct_file = f"../data/lct/lct_{self.width}_{self.height}_{left}_{bottom}_{right}_{top}.npy"
        if not os.path.isfile(lct_file) or not use_cache:
            lc_grid = np.zeros((self.width, self.height), np.int16)
            for y in range(0, self.height):
                for x in range(0, self.width):
                    map_x = left + (right - left) * y / self.height
                    map_y = bottom + (top - bottom) * (1 - x / self.width)
                    map_value = land_cover_data.cx[map_x:map_x + 0.0001, map_y:map_y + 0.0001]
                    slice_length = len(map_value)
                    if slice_length > 0:
                        cell_type = map_value.Code_18.values[0]
                    else:
                        cell_type = 999
                    lc_grid[x, y] = cell_type
                # print(f"{y + 1} / {self.height}")
            np.save(lct_file, lc_grid)

        grid = np.load(lct_file)
        plot_grid = False
        if plot_grid:
            plt.imshow(grid, interpolation='nearest')
            plt.title("Land cover grid")
            plt.show()

        land_cover_types = [int(x) for x in list(land_cover_rates.keys())]
        land_cover_types.sort()
        for y in range(0, self.height):
            for x in range(0, self.width):
                grid[x, y] = land_cover_types.index(grid[x, y])
        land_cover_rates = [x[1] for x in sorted(list(land_cover_rates.items()), key=lambda l: l[0])]
        return np.array(land_cover_rates), grid

    def weather_grid(self, weather_data):
        # Start datetime from requests/weather.py
        weather_start_time = datetime(year=2020, month=7, day=1)
        hour_diff = self.coord_bounds['time_start'] - weather_start_time
        slice_start = hour_diff.days * 24
        slice_length = (self.coord_bounds['time_end'] - self.coord_bounds['time_start']).days * 24

        grids = {}
        for key in weather_data:
            sliced = weather_data[key][slice_start:slice_start + slice_length, :, :]
            sliced = np.swapaxes(sliced, 1, 2)
            if sliced.shape[0] == 0:
                return None
            # slices_resized = np.zeros((self.duration, self.width, self.height))
            temp_vol = np.zeros((self.width, self.height, sliced.shape[0]), dtype=np.float64)
            for hour in range(sliced.shape[0]):
                temp_vol[:, :, hour] = cv2.resize(sliced[hour, :, :], dsize=(self.height, self.width))
            volume = np.zeros((self.width, self.height, self.duration), dtype=np.float64)
            for x in range(self.width):
                volume[x, :, :] = cv2.resize(temp_vol[x, :, :], dsize=(self.duration, self.height))
            grids[key] = volume

        weather = np.zeros((self.width, self.height, self.duration, 2), dtype=np.float64)
        weather[:, :, :, 0] = grids['u10']
        weather[:, :, :, 1] = grids['v10']
        return weather

    def height_grid(self, nc):
        grid = cv2.resize(nc, dsize=(self.width, self.height))
        return grid

    def fire_grid(self, gdf):
        # gdf.plot()
        # plt.show()
        sorted_gdf = gdf.sort_values(by=['FDate'])
        grid = np.zeros((self.width, self.height, gdf.shape[0]), dtype=np.bool)
        for areaIndex in range(gdf.shape[0]):
            area = sorted_gdf['geometry'][areaIndex]
            for x in range(self.width):
                for y in range(self.height):
                    lon = self.coord_bounds['left'] + x * self.spatial_resolution
                    lat = self.coord_bounds['bottom'] + y * self.spatial_resolution
                    is_in_polygon = Point(lon, lat).within(area)
                    grid[x, y, areaIndex] = is_in_polygon

        plot_fire_grid = False
        if plot_fire_grid:
            for i in range(grid.shape[2]):
                plt.imshow(grid[:, :, i], interpolation='nearest')
                plt.title(f"Fire shape [{i + 1}/{grid.shape[2]}]")
                plt.show()
        return grid
