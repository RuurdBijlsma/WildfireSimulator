from load_all import load_all
import cuda_python

land_cover_grid, land_cover_rates, height_grid, fire_grid, weather_grid, params = load_all()
result = cuda_python.batch_simulate(land_cover_grid, land_cover_rates, height_grid, fire_grid, weather_grid, params)
print("RESULT", result)
