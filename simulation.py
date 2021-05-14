import math
import os

import geopandas
import numpy as np
from matplotlib import pyplot as plt

# TODO
# Change units to real units
# Wind speed to m/s
# Height to meters
# Spread rate: 1 / time to burn 1 cell (seconds^-1)
# Use real world data!
# 0. Use real-world units (1 cell is 50x50 meter or something)
# 1. pick bbox
# 2. pick (land cover/wind/height) for each cell
# 3. base simulation stuff on data
# Consider crown/bush/sub-ground/surface fires
# * Maybe have a 3d CA that is 2/3 layers high?
# Consider high intensity fires can jump larger distances than others

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
        self.wind_matrix = np.ones((3, 3), np.float)
        for x in range(3):
            for y in range(3):
                # Center = (1, 1)
                rel_x = x - 1
                rel_y = y - 1
                self.wind_matrix[x, y] = rel_x * wind_from_x + rel_y * wind_from_y
        self.wind_matrix = np.clip((self.wind_matrix * wind_speed / 2 + 1), a_min=0, a_max=3)

        self.width = 50
        self.height = 50
        # features: [fire activity, fuel, height] (later land cover at the end)
        self.num_features = 3
        self.grid = np.zeros((self.width, self.height, self.num_features))
        # Spread rate
        self.grid[:, :, 1] = 1
        # height
        self.grid[:, :, 2] = 0
        self.grid[:, :, 2] = np.array(range(self.width)) / self.width
        # self.grid[0:15, 0:15, 2] = 1
        # self.grid[:, :, 2] = np.random.random((self.width, self.height))
        # Start fire activity at 0,0
        self.grid[20:23, 20:23, 0] = 1

        # Remove fuel from area
        self.grid[self.width // 2, 0:self.height // 2, 1] = 0

    def load_land_cover(self):
        p = os.path.abspath("data/DATA/U2018_CLC2018_V2020_20u1.gpkg")
        print(p)
        gdf_mask = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
        # gdf = geopandas.read_file(p, mask=gdf_mask[gdf_mask.name == "Cyprus"])
        gdf = geopandas.read_file(p, bbox=(1.6 * 1e6, 1.7 * 1e6, 6.3 * 1e6, 6.5 * 1e6))

        print(gdf.head())
        gdf.plot("Code_18")
        plt.show()
        print("DONE")

        return gdf

    def tick(self):
        new_grid = self.grid.copy()
        for x in np.arange(1, self.width - 1, dtype=np.int):
            for y in np.arange(1, self.height - 1, dtype=np.int):
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
                height_diff_matrix = (cell[2] - neighbours[:, :, 2]) * self.height_multiplier + 1
                activity_matrix *= height_diff_matrix
                activity = activity_matrix.mean()
                # If neighbouring fire activity is high enough
                if activity + 0.2 > np.random.random():
                    # Increase fire activity in current cell
                    new_grid[x, y, 0] += cell[1] * activity
        self.grid = new_grid
