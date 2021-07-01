import numpy as np
import pyswarms as ps
import csv

from matplotlib import pyplot as plt
from pyswarms.utils import plotters

from simulation import Simulation

sim = Simulation(show_plots=False)
ticks_to_end = round(sim.time_between_burnt_areas / sim.time_per_tick)
print(f"Simulating {ticks_to_end} ticks")


def get_fitness(elements, i):
    sim.show_plots = i == 0
    sim.set_parameters(elements)
    sim.reset_grid()
    for i in range(ticks_to_end):
        sim.tick()
    return sim.get_fitness()


def f(x):
    n_particles = x.shape[0]
    j = [get_fitness(x[i], i) for i in range(n_particles)]
    return np.array(j)


land_cover_rates = []
with open('landCoverSpreadRate.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    for [clc_code, _, spread_rate] in reader:
        if clc_code == 'CLC_CODE':
            continue
        land_cover_rates.append(float(spread_rate))
land_cover_rates = np.array(land_cover_rates)

# c1 :float # cognitive parameter
# c2 :float # social parameter
# w :float # inertia parameter
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
# options = {'c1': 0.7, 'c2': 0.3, 'w': 0.9}
# options = {'c1': 0.3, 'c2': 0.3, 'w': 0.9}
# options = {'c1': 0.5, 'c2': 0.1, 'w': 0.9}
# options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
# options = {'c1': 0.5, 'c2': 0.3, 'w': 0.7}
# options = {'c1': 0.5, 'c2': 0.3, 'w': 1.2}
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

# Create a RandomSearch object
# n_selection_iters is the number of iterations to run the searcher
# iters is the number of iterations to run the optimizer
# g = RandomSearch(ps.single.LocalBestPSO, n_particles=40,
#                  dimensions=20, options=options, objective_func=f,
#                  iters=10, n_selection_iters=100)
#
# best_score, best_options = g.search()
# print(best_score, best_options)
