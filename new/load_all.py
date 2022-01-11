from load_fire import load_fire_gdf, gdf_to_bounds, bounds_to_square, parse_fire, get_fire_lists, get_train_test_split
from load_weather import load_weather
from load_height import load_height
from bounds_restrictions import get_restrictions
from grid import Grid
from load_land_cover import load_land_cover
# from utils import plot_bounds
from sklearn.model_selection import KFold
from constants import seed


class FireLoader:
    def __init__(self, max_data=1000000, n_splits=5):
        # get data bounds restrictions
        self.bounds_restrictions = get_restrictions()
        # plot_bounds(self.bounds_restrictions)
        self.fire_lists = get_fire_lists(self.bounds_restrictions)[0:max_data]
        # Data must be multiple of `n_splits`, to get same size train/test splits every time
        remainder = len(self.fire_lists) % n_splits
        if remainder != 0:
            self.fire_lists = self.fire_lists[0:len(self.fire_lists) - remainder]
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Get random fire
    @staticmethod
    def load_fire(fire_tuple):
        fire = parse_fire(fire_tuple[0], fire_tuple[1])
        # get bounds of fire
        fire_gdf = load_fire_gdf(fire)
        if fire_gdf.shape[0] == 0 or fire_gdf.shape[1] != 5:
            print("WARNING fire gdf not loaded properly, skipping fire")
            return None, None, None, None, None

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
            return None, None, None, None, None

        grid = Grid(bounds)
        if grid.duration == 0 or grid.width == 0 or grid.height == 0:
            print("WARNING, fire bounds smaller than simulation resolution! SKIP!")
            return None, None, None, None, None

        weather_grid = grid.weather_grid(weather_data)
        if weather_grid is None:
            print("WARNING! weather data not available for fire! SKIP!")
            return None, None, None, None, None
        height_grid = grid.height_grid(height_data)
        fire_grid = grid.fire_grid(fire_gdf)
        land_cover_rates, land_cover_grid = grid.land_cover_grid(land_cover_data)
        # print(fire_gdf, weather_data, height_data)

        return land_cover_grid, land_cover_rates, height_grid, fire_grid, weather_grid
