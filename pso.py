import numpy as np
import pyswarms as ps
from pyswarms.utils.search import RandomSearch
from pyswarms.utils.functions import single_obj as fx

# Set-up choices for the parameters
options = {
    'c1': (1, 5),
    'c2': (6, 10),
    'w': (2, 5),
    'k': (11, 15),
    'p': 1
}

# Create a RandomSearch object
# n_selection_iters is the number of iterations to run the searcher
# iters is the number of iterations to run the optimizer
g = RandomSearch(ps.single.LocalBestPSO, n_particles=40,
                 dimensions=20, options=options, objective_func=fx.sphere,
                 iters=10, n_selection_iters=100)

best_score, best_options = g.search()
print(best_score, best_options)
