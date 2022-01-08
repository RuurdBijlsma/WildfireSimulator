import numpy as np
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils import plotters
from load_all import FireLoader
from sklearn import metrics
import cuda_python
from sklearn.model_selection import KFold


class PSO:
    show_plots = False

    def __init__(self):
        kf = KFold(n_splits=5)
        fires = [1, 2, 3, 4, 5, 6]
        for train, test in kf.split(fires):
            print(train, test)
        # print(train, test)

        self.loader = FireLoader()
        self.land_cover_grid, self.land_cover_rates, self.height_grid, self.fire_grid, \
        self.weather_grid, self.initial_params = self.loader.load_fire(self.loader.train[0])

    def get_fitness(self, lcr, params):
        result = cuda_python.batch_simulate(self.land_cover_grid, lcr, self.height_grid,
                                            self.fire_grid, self.weather_grid,
                                            params)
        # Compare burnt area result with self.burnt_area_end
        simulated_burnt_area = (1 - result) > 0.8
        burnt_area = self.fire_grid[:, :, self.fire_grid.shape[2] - 1] > 0.8
        flat_ba = burnt_area.flatten()
        auc = np.zeros(result.shape[2])
        for i in range(result.shape[2]):
            flat_sba = simulated_burnt_area[:, :, i].flatten()
            auc[i] = metrics.roc_auc_score(flat_ba, flat_sba)

            if self.show_plots:
                fpr, tpr, threshold = metrics.roc_curve(flat_ba, flat_sba)
                plt.plot(fpr, tpr, linestyle='--', label='ROC Curve')
                plt.title(f"AUC: {round(auc[i] * 100) / 100}")
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.legend()
                plt.show()

                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(simulated_burnt_area[:, :, i], interpolation='nearest')
                axs[0].set_title(f"sim burnt area, AUC: {round(auc[i] * 100) / 100}")
                axs[1].imshow(burnt_area, interpolation='nearest')
                axs[1].set_title('actual burnt area')
                plt.show()

        return 1 - auc

    def optimize(self):
        def wrap_self(inst):
            def f(x):
                lcr = x[:, 0:len(inst.land_cover_rates)].transpose()
                params = x[:, len(inst.land_cover_rates):].transpose()
                return inst.get_fitness(lcr, params)

            return f

        options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
        swarm_size = 20
        dimensions = self.land_cover_rates.shape[0] + len(self.initial_params)
        max_bound = 5 * np.ones(dimensions)
        min_bound = np.zeros(dimensions)
        bounds = (min_bound, max_bound)
        initial_values = np.zeros((swarm_size, dimensions))
        initial_values[:, 0:len(self.land_cover_rates)] = self.land_cover_rates
        initial_values[:, len(self.land_cover_rates):] = self.initial_params
        optimizer = ps.single.GlobalBestPSO(n_particles=swarm_size,
                                            init_pos=initial_values,
                                            dimensions=dimensions,
                                            options=options,
                                            bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(wrap_self(self), iters=100)
        plotters.plot_cost_history(cost_history=optimizer.cost_history)
        plt.show()
        print(pos)


if __name__ == "__main__":
    pso = PSO()
    pso.optimize()
