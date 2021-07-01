import geopandas
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point
from math import sin, cos, radians
from simulation_utils import load_burnt_areas, load_land_cover, distance_between_coordinates
from regions import region
from sklearn import metrics


# TODO Only optimize land cover spread rates for land cover that appears in region
# PSO
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
# Setting start situation from burnt area data is not accurate because i set it to fire activity


class Simulation:
    def __init__(self, show_plots=True):
        self.spread_params = {
            "burn_rate": 0.1,
            "activity_threshold": .2,
            "death_threshold": 0.1,
            "death_rate": .2,
            "area_effect_multiplier": 1,
            "height_effect_multiplier": 2,
            "spread_speed": 1.5,
        }

        self.show_plots = show_plots

        self.time_per_tick = 60 * 60  # 1 hour
        self.time_between_burnt_areas = 60 * 60 * 24 * 10  # 10 days
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

        # Direction:
        #   0: wind from south
        #  90: wind from east
        # 180: wind from north
        # 270: wind from west
        wind_direction = radians(225)
        # 0 speed because wind isn't in data yet
        wind_speed = 0
        wind_from_x = cos(wind_direction)
        wind_from_y = sin(wind_direction)
        self.wind_matrix = np.ones((3, 3), np.float32)
        for x in range(3):
            for y in range(3):
                # Center = (1, 1)
                rel_x = x - 1
                rel_y = y - 1
                self.wind_matrix[x, y] = rel_x * wind_from_x + rel_y * wind_from_y
        self.wind_matrix = np.clip((self.wind_matrix * wind_speed / 2 + 1), a_min=0, a_max=3)

        self.width = 25
        self.height = 25
        # features: [fire activity, fuel, land_cover, height]
        self.num_features = 4

        # Land cover
        # Mercator x/y bounds

        bounds = region["bounds"]
        bounds['Coordinates'] = list(zip(bounds.Longitude, bounds.Latitude))
        bounds['Coordinates'] = bounds['Coordinates'].apply(Point)
        frame = geopandas.GeoDataFrame(bounds, geometry='Coordinates')
        frame.set_crs(epsg=4326, inplace=True)

        self.burnt_area_start, self.burnt_area_end = load_burnt_areas(frame, self.width, self.height)
        self.land_cover_rates, self.land_cover_types = load_land_cover(frame, self.width, self.height)

        self.cell_width = distance_between_coordinates(
            bounds.Latitude[0], bounds.Longitude[0],
            bounds.Latitude[0], bounds.Longitude[1],
        ) / self.width
        self.cell_height = distance_between_coordinates(
            bounds.Latitude[0], bounds.Longitude[0],
            bounds.Latitude[1], bounds.Longitude[0],
        ) / self.height
        self.cell_area = self.cell_width * self.cell_height

        self.reset_grid()

    def set_parameters(self, parameters):
        # Last N params in `parameters` are for self.spread_params
        # TODO add wind and other params stuff
        for index, key in enumerate(self.land_cover_rates):
            self.land_cover_rates[key] = parameters[index]
        for index, key in enumerate(self.spread_params):
            self.spread_params[key] = parameters[index + len(self.land_cover_rates)]

    def reset_grid(self):
        self.grid = np.zeros((self.width, self.height, self.num_features))
        # Spread rate
        self.grid[:, :, 1] = 1
        # height
        self.grid[:, :, 3] = 0
        # Start fire activity at 0,0
        # self.grid[20:23, 20:23, 0] = 1
        # Start fire activity at burnt area data
        self.grid[:, :, 0] = np.clip(self.burnt_area_start, 0, 1)

        # Remove fuel from area
        # self.grid[self.width // 2, 0:self.height // 2, 1] = 0

        # Set spread rate based on land cover type
        for y in range(0, self.height):
            for x in range(0, self.width):
                cell_type = self.land_cover_types[x, y]
                # edit land_Cover_rates with pso
                cell_spread_rate = self.land_cover_rates[str(cell_type)]
                # print(cell_spread_rate)
                self.grid[x, y, 2] = cell_spread_rate
        print("GRID RESET")

    def get_fitness(self):
        # Compare burnt area result with self.burnt_area_end
        simulated_burnt_area = (1 - self.grid[:, :, 1]) > 0.8
        burnt_area = self.burnt_area_end > 0.8
        flat_ba = burnt_area.flatten()
        flat_sba = simulated_burnt_area.flatten()
        auc = metrics.roc_auc_score(flat_ba, flat_sba)

        if self.show_plots:
            fpr, tpr, threshold = metrics.roc_curve(flat_ba, flat_sba)
            plt.plot(fpr, tpr, linestyle='--', label='ROC Curve')
            plt.title(f"AUC: {round(auc * 100) / 100}")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.show()

            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(simulated_burnt_area, interpolation='nearest')
            axs[0].set_title(f"sim burnt area, AUC: {round(auc * 100) / 100}")
            axs[1].imshow(burnt_area, interpolation='nearest')
            axs[1].set_title('actual burnt area')
            plt.show()

        return 1 - auc

    def tick(self):
        new_grid = self.grid.copy()
        for x in np.arange(1, self.width - 1, dtype=np.int32):
            for y in np.arange(1, self.height - 1, dtype=np.int32):
                cell = self.grid[x, y, :]
                # If cell has fire activity
                if cell[0] > 0:
                    # Remove fire activity amount from fuel
                    new_grid[x, y, 1] = cell[1] - cell[0] * self.spread_params["burn_rate"]
                # If fuel ran out, set fire activity and fuel to 0
                if new_grid[x, y, 1] < 0:
                    new_grid[x, y, 1] = 0
                    new_grid[x, y, 0] = 0
                # Get mean fire activity around cell
                # Multiply neighbours with vector, wind from north would be:
                # [1.5, 1.5, 1.5]
                # [1,   0,   1  ]
                # [0.5, 0.5, 0.5]
                # Also consider height difference and land cover
                neighbours = self.grid[x - 1:x + 2, y - 1:y + 2, :]
                # Fire activity from neighbour cell counts more if wind comes from there
                activity_matrix = np.multiply(neighbours[:, :, 0], self.wind_matrix)
                # Same but for height, going down decreases activity spread, going up increases it
                height_diff_matrix = (cell[3] - neighbours[:, :, 3]) * self.spread_params[
                    "height_effect_multiplier"] + 1
                activity_matrix *= height_diff_matrix
                # Mean of activity matrix times spread rate based on land cover of current cell
                activity = activity_matrix.mean() * cell[2]
                # If neighbouring fire activity is high enough
                if activity > self.spread_params["activity_threshold"] + np.random.random() / 5:
                    # Increase fire activity in current cell
                    new_grid[x, y, 0] += cell[1] * activity / \
                                         (self.cell_area / self.spread_params["spread_speed"] * self.spread_params[
                                             "area_effect_multiplier"])
                elif activity <= self.spread_params["death_threshold"]:
                    new_grid[x, y, 0] /= 1 + (self.spread_params["death_rate"] /
                                              (self.cell_area * self.spread_params["area_effect_multiplier"]))
        self.grid = new_grid


if __name__ == '__main__':
    sim = Simulation(show_plots=True)
    ticks_to_end = round(sim.time_between_burnt_areas / sim.time_per_tick)
    # ticks_to_end = 40
    print(f"Simulating {ticks_to_end} ticks")
    for i in range(ticks_to_end):
        sim.tick()
    print(sim.get_fitness())
