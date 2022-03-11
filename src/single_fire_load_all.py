from load_fire import load_fire_gdf, gdf_to_bounds, bounds_to_square, parse_fire, get_fire_lists, get_train_test_split
from load_weather import load_weather
from load_height import load_height
from bounds_restrictions import get_restrictions
from grid import Grid
from load_land_cover import load_land_cover
# from utils import plot_bounds
from sklearn.model_selection import KFold
from constants import seed
import numpy as np


class FireLoader2:
    def __init__(self, max_data=1000000, n_splits=5):
        # get data bounds restrictions
        self.bounds_restrictions = get_restrictions()
        # plot_bounds(self.bounds_restrictions)
        self.fire_lists = get_fire_lists(self.bounds_restrictions)[0:max_data]

    # Get random fire
    @staticmethod
    def load_fire(fire_tuple):
        fire = parse_fire(fire_tuple[0], fire_tuple[1])
        # get bounds of fire
        fire_gdf = load_fire_gdf(fire)
        # fire gdf has to be more than 5 time steps for sufficient train/test samples
        if fire_gdf.shape[0] <= 5 or fire_gdf.shape[1] != 5:
            print("WARNING fire gdf not loaded properly, skipping fire")
            return None, None

        bounds = bounds_to_square(gdf_to_bounds(fire_gdf))
        land_cover_data = load_land_cover(bounds)
        # get data
        weather_data = load_weather(bounds)

        height_data = load_height(bounds)

        if height_data.shape[0] == 0 or height_data.shape[1] == 0 \
                or land_cover_data.shape[0] == 0 or land_cover_data.shape[1] == 0 \
                or weather_data['u10'].shape[0] == 0 or weather_data['u10'].shape[1] == 0 \
                or weather_data['u10'].shape[2] == 0 or weather_data['v10'].shape[0] == 0 \
                or weather_data['v10'].shape[1] == 0 or weather_data['v10'].shape[2] == 0:
            print("WARNING, no data for this fire! SKIP!")
            return None, None

        grid = Grid(bounds)
        if grid.duration == 0 or grid.width == 0 or grid.height == 0:
            print("WARNING, fire bounds smaller than simulation resolution! SKIP!")
            return None, None

        weather_grid = grid.weather_grid(weather_data)
        if weather_grid is None:
            print("WARNING! weather data not available for fire! SKIP!")
            return None, None
        height_grid = grid.height_grid(height_data)
        fire_grid = grid.fire_grid(fire_gdf)
        land_cover_rates, land_cover_grid = grid.land_cover_grid(land_cover_data)
        # print(fire_gdf, weather_data, height_data)

        train_percentage = .7
        fire_timesteps = fire_grid.shape[2]
        train_end_fire_index = round(fire_timesteps * train_percentage)
        train_start = fire_grid[:, :, 0]
        train_end = fire_grid[:, :, 1]
        for i in range(0, fire_timesteps - train_end_fire_index):
            train_end = np.logical_or(train_end, fire_grid[:, :, i])
        test_start = np.copy(train_end)
        test_end = fire_grid[:, :, fire_timesteps - 1]

        train_fire = np.dstack((train_start, train_end))
        test_fire = np.dstack((test_start, test_end))

        weather_start_index = 0
        weather_end_index = weather_grid.shape[2]
        weather_mid_index = round(weather_end_index * train_percentage)
        train_weather = weather_grid[:, :, weather_start_index:weather_mid_index, :]
        test_weather = weather_grid[:, :, weather_mid_index:weather_end_index, :]

        train = (land_cover_grid, land_cover_rates, height_grid, train_fire, train_weather)
        test = (land_cover_grid, land_cover_rates, height_grid, test_fire, test_weather)
        return train, test
