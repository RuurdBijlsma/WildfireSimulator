import numpy as np
import pyswarms as ps
import csv

from simulation import Simulation

simulation = Simulation()


def get_fitness(elements):
    return elements.var()


def f(x):
    n_particles = x.shape[0]
    j = [get_fitness(x[i]) for i in range(n_particles)]
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
dimensions = land_cover_rates.shape[0]
max_bound = 2 * np.ones(dimensions)
min_bound = np.zeros(dimensions)
bounds = (min_bound, max_bound)
optimizer = ps.single.GlobalBestPSO(n_particles=100,
                                    dimensions=dimensions,
                                    options=options,
                                    bounds=bounds)

# Perform optimization
cost, pos = optimizer.optimize(f, iters=1000)
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
