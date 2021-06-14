import math
import os

import geopandas
import numpy as np
from matplotlib import pyplot as plt
import csv
import matplotlib.patches as patches
from shapely.geometry import Point
import pandas as pd


# TODO
# need results
# for that i need
# to compare stuff
# t test ?
# i need evaluate simulation
# compare iou with real data
# i need to read burnt data

# PSO
# Learn how to use?
# Let pso set arguments for spread rate data
# Give rates object to simulation
# Do the pso thing

# Spread rate: 1 / time to burn 1 cell (seconds^-1)
# 0. Use real-world units (1 cell is 50x50 meter or something)
# 1. pick bbox
# 2. pick (land cover/wind/height) for each cell
# 3. base simulation stuff on data
# Consider crown/bush/sub-ground/surface fires
# * Maybe have a 3d CA that is 2/3 layers high?
# Consider high intensity fires can jump larger distances than others

def load_land_cover(frame_bbox):
    p = os.path.abspath("data/DATA/U2018_CLC2018_V2020_20u1.gpkg")
    print(p)
    # gdf_mask = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    # gdf_mask.plot()
    # gdf = geopandas.read_file(p, mask=gdf_mask[gdf_mask.name == "Cyprus"])
    gdf = geopandas.read_file(p, bbox=frame_bbox)
    gdf.to_crs(epsg=4326, inplace=True)

    # gdf = gdf.to_crs("EPSG:4326")

    print(gdf.head())
    [left, bottom, right, top] = frame_bbox.total_bounds
    rect = patches.Rectangle((left, bottom), (right - left), (top - bottom),
                             linewidth=1, edgecolor='r', facecolor='none')
    fig, ax = plt.subplots()
    gdf.plot("Code_18", ax=ax)
    ax.add_patch(rect)
    plt.show()
    print("DONE")

    return gdf


class Simulation:
    def __init__(self):
        self.L = 1  # Cell is 1 meter by 1 meter
        self.cell_area = 1  # 1 * 1
        # new cell state is old cell state + sum of all (neighbours mju value * neighbours state)
        # round cell state to 0..0.1...1
        # Per cell parameter:
        # burnt fraction: burnt area / area of cell
        # spread rate: time needed for this cell to burn out completely (depends on ground type)
        # wind speed: ?
        # height: ?
        # Size of the discrete time step:
        # ~t = 1 / R
        # R = max(cell_spread_rates, 1 <= x <= width, 1 <= y <= height)

        self.height_multiplier = 3
        # Direction:
        #   0: wind from south
        #  90: wind from east
        # 180: wind from north
        # 270: wind from west
        wind_direction = math.radians(225)
        wind_speed = 0
        wind_from_x = math.cos(wind_direction)
        wind_from_y = math.sin(wind_direction)
        self.wind_matrix = np.ones((3, 3), np.float32)
        for x in range(3):
            for y in range(3):
                # Center = (1, 1)
                rel_x = x - 1
                rel_y = y - 1
                self.wind_matrix[x, y] = rel_x * wind_from_x + rel_y * wind_from_y
        self.wind_matrix = np.clip((self.wind_matrix * wind_speed / 2 + 1), a_min=0, a_max=3)

        self.width = 70
        self.height = 50
        # features: [fire activity, fuel, land_cover, height]
        self.num_features = 4

        # Land cover
        # Mercator x/y bounds
        self.land_cover_rates = {}
        with open('landCoverSpreadRate.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for [clc_code, _, spread_rate] in reader:
                if clc_code == 'CLC_CODE':
                    continue
                self.land_cover_rates[clc_code] = float(spread_rate)

        df = pd.DataFrame({
            'Latitude': [34.7, 35.2],  # Y value
            'Longitude': [32.9, 33.9]  # X value
        })
        df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
        df['Coordinates'] = df['Coordinates'].apply(Point)
        frame = geopandas.GeoDataFrame(df, geometry='Coordinates')
        frame.set_crs(epsg=4326, inplace=True)
        [left, bottom, right, top] = frame.total_bounds

        self.gdf = load_land_cover(frame)
        self.land_cover_types = np.zeros((self.width, self.height), np.int16)
        for y in range(0, self.height):
            for x in range(0, self.width):
                map_x = left + (right - left) * y / self.height
                map_y = bottom + (top - bottom) * (1 - x / self.width)
                map_value = self.gdf.cx[map_x:map_x + 0.0001, map_y:map_y + 0.0001]
                slice_length = len(map_value)
                if slice_length > 0:
                    cell_type = map_value.Code_18.values[0]
                else:
                    cell_type = 999
                self.land_cover_types[x, y] = cell_type
            print(f"{y + 1} / {self.height}")

        self.reset_grid()

    def reset_grid(self):
        self.grid = np.zeros((self.width, self.height, self.num_features))
        # Spread rate
        self.grid[:, :, 1] = 1
        # height
        self.grid[:, :, 3] = 0
        # Set height to gradient
        self.grid[:, :, 3] = np.array(range(self.height)) / self.height
        # self.grid[0:15, 0:15, 2] = 1
        # self.grid[:, :, 2] = np.random.random((self.width, self.height))
        # Start fire activity at 0,0
        self.grid[20:23, 20:23, 0] = 1

        # Remove fuel from area
        self.grid[self.width // 2, 0:self.height // 2, 1] = 0

        # Set spread rate based on land cover type
        for y in range(0, self.height):
            for x in range(0, self.width):
                cell_type = self.land_cover_types[x, y]
                # edit land_Cover_rates with pso
                cell_spread_rate = self.land_cover_rates[str(cell_type)]
                # print(cell_spread_rate)
                self.grid[x, y, 2] = cell_spread_rate
        print("GRID RESET DONE")

    def get_fitness(self):
        return np.random.random()

    def tick(self):
        new_grid = self.grid.copy()
        for x in np.arange(1, self.width - 1, dtype=np.int32):
            for y in np.arange(1, self.height - 1, dtype=np.int32):
                cell = self.grid[x, y, :]
                # If cell has fire activity
                if cell[0] > 0:
                    # Remove fire activity amount from fuel
                    new_grid[x, y, 1] = cell[1] - cell[0]
                if new_grid[x, y, 1] < 0:
                    new_grid[x, y, 1] = 0
                    new_grid[x, y, 0] = 0
                # Get mean fire activity around cell
                # Multiply neighbours with mju vector (wind from north would be:
                # [1.5, 1.5, 1.5]
                # [1,   0,   1  ]
                # [0.5, 0.5, 0.5]
                # Also consider height difference and later land cover
                neighbours = self.grid[x - 1:x + 2, y - 1:y + 2, :]
                activity_matrix = np.multiply(neighbours[:, :, 0], self.wind_matrix)
                height_diff_matrix = (cell[3] - neighbours[:, :, 3]) * self.height_multiplier + 1
                activity_matrix *= height_diff_matrix
                # Mean of activity matrix times spread rate based on land cover of current cell
                activity = activity_matrix.mean() * cell[2]
                # If neighbouring fire activity is high enough
                if activity + 0.2 > np.random.random():
                    # Increase fire activity in current cell
                    new_grid[x, y, 0] += cell[1] * activity
        self.grid = new_grid
