import math
from datetime import timedelta
import numpy as np
from shapely.geometry import Point, Polygon
import cv2
from datetime import datetime


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
        self.duration = self.timedelta / self.temporal_resolution

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
            slices_resized = np.zeros((sliced.shape[0], self.width, self.height))
            for hour in range(sliced.shape[0]):
                slices_resized[hour, :, :] = cv2.resize(sliced[hour, :, :], dsize=(self.width, self.height))
            grids[key] = slices_resized

        return grids

    def height_grid(self, nc):
        grid = cv2.resize(nc, dsize=(self.width, self.height))
        return grid

    def fire_grid(self, gdf):
        sorted_gdf = gdf.sort_values(by=['FDate'])
        grid = np.zeros((self.width, self.height, gdf.shape[1]), dtype=np.bool)
        for areaIndex in range(gdf.shape[1]):
            area = sorted_gdf['geometry'][areaIndex]
            for x in range(self.width):
                for y in range(self.height):
                    lon = self.coord_bounds['left'] + x * self.spatial_resolution
                    lat = self.coord_bounds['bottom'] + y * self.spatial_resolution
                    isInPolygon = Point(lon, lat).within(area)
                    grid[x, y, areaIndex] = isInPolygon

        return grid
