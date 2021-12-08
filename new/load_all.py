from load_fire import load_fire_gdf, gdf_to_bounds, bounds_to_square, pick_fire
from load_weather import load_weather
from load_height import load_height
from bounds_restrictions import get_restrictions
from grid import Grid
from load_land_cover import load_land_cover
from utils import plot_bounds

# get data bounds restrictions
bounds_restrictions = get_restrictions()
plot_bounds(bounds_restrictions)

# choose fire

fire = pick_fire(bounds_restrictions)
# get bounds of fire
fire_gdf = load_fire_gdf(fire)
bounds = bounds_to_square(gdf_to_bounds(fire_gdf))
# get data
land_cover_data = load_land_cover(bounds)
weather_data = load_weather(bounds)
height_data = load_height(bounds)

grid = Grid(bounds)
weather_grid = grid.weather_grid(weather_data)
height_grid = grid.height_grid(height_data)
land_cover_rates, land_cover_grid = grid.land_cover_grid(land_cover_data)
fire_grid = grid.fire_grid(fire_gdf)

print(fire_gdf, weather_data, height_data)
# TODO:
# PSO reimplementation
# Call cuda sim from here
# Send grids to cuda sim
# Send PSO params to cuda sim
# Get result from cuda sim (std in/out)
