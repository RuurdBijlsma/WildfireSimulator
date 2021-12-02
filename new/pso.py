import numpy as np
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils import plotters


def optimize(weather_grid, fire_grid, height_grid, land_cover_grid, land_cover_rates):
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    swarm_size = 20
    dimensions = land_cover_rates.shape[0] + len(sim.spread_params)
    max_bound = 2 * np.ones(dimensions)
    min_bound = np.zeros(dimensions)
    bounds = (min_bound, max_bound)
    initial_values = np.zeros((swarm_size, dimensions))
    initial_values[:, 0:len(land_cover_rates)] = land_cover_rates
    initial_spread_params = np.array(list(sim.spread_params.values()))
    initial_values[:, len(land_cover_rates):] = initial_spread_params
    optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                        init_pos=initial_values,
                                        dimensions=dimensions,
                                        options=options,
                                        bounds=bounds)

    # Perform optimization
    cost, pos = optimizer.optimize(f, iters=75)
    plotters.plot_cost_history(cost_history=optimizer.cost_history)
    plt.show()
    print(pos)
