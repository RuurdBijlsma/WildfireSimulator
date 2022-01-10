from load_fire import load_fire_gdf, gdf_to_bounds, bounds_to_square, parse_fire, get_fire_lists, get_train_test_split
from load_weather import load_weather
from load_height import load_height
from bounds_restrictions import get_restrictions
from grid import Grid
from load_land_cover import load_land_cover
from utils import plot_bounds
from sklearn.model_selection import KFold
from constants import seed


class FireLoader:
    def __init__(self, max_data=1000000, n_splits=5):
        # get data bounds restrictions
        self.bounds_restrictions = get_restrictions()
        # plot_bounds(self.bounds_restrictions)
        self.fire_lists = get_fire_lists(self.bounds_restrictions)[0:max_data]
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Get random fire
    @staticmethod
    def load_fire(fire_tuple):
        fire = parse_fire(fire_tuple[0], fire_tuple[1])
        # get bounds of fire
        fire_gdf = load_fire_gdf(fire)
        bounds = bounds_to_square(gdf_to_bounds(fire_gdf))
        land_cover_data = load_land_cover(bounds)
        # get data
        weather_data = load_weather(bounds)
        height_data = load_height(bounds)

        grid = Grid(bounds)
        weather_grid = grid.weather_grid(weather_data)
        height_grid = grid.height_grid(height_data)
        fire_grid = grid.fire_grid(fire_gdf)
        land_cover_rates, land_cover_grid = grid.land_cover_grid(land_cover_data)

        # print(fire_gdf, weather_data, height_data)

        return land_cover_grid, land_cover_rates, height_grid, fire_grid, weather_grid
        # TODO:
        # PSO reimplementation
        # Call cuda sim from here
        # Send grids to cuda sim
        # Send PSO params to cuda sim
        # Get result from cuda sim (std in/out)
