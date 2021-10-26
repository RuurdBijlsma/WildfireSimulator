from load_fire import load_fire_gdf, gdf_to_bounds, bounds_to_square, pick_fire_id
from load_weather import load_weather
from load_height import load_height
from bounds_restrictions import get_restrictions

# get data bounds restrictions
bounds_restrictions = get_restrictions()
# choose fire
fire_id = pick_fire_id(bounds_restrictions)
# get bounds of fire
fire_gdf = load_fire_gdf(fire_id)
bounds = bounds_to_square(gdf_to_bounds(fire_gdf))
# get data
weather_data = load_weather(bounds)
height_data = load_height(bounds)

print(fire_gdf, weather_data, height_data)
