import numpy as np
import pyswarms as ps
from matplotlib import pyplot as plt
from pyswarms.utils.plotters import (plot_cost_history, plot_contour)
from single_fire_load_all import FireLoader2
from sklearn import metrics
import cv2
import time
import cuda_python
from statistics import mean


class PSO:
    show_plots = False
    # PSO Options
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
    # oh_strategy = {"w": 'exp_decay', "c1": 'nonlin_mod', "c2": 'lin_variation'}
    swarm_size = 30
    iterations = 30
    data_size = 7777

    best_auc_for_plot = 0

    def __init__(self):
        self.loader = FireLoader2(max_data=self.data_size)
        self.land_cover_grid = None
        self.height_grid = None
        self.fire_grid = None
        self.weather_grid = None

    def full_train(self):
        params = np.array([.3, .1, .1, 2, .1, 2, .2, 1, .1, 1])
        land_cover_rates = None
        training_costs = []
        test_costs = []
        for i, fire in enumerate(self.loader.fire_lists):
            self.best_auc_for_plot = 0
            print(f"=========================== NOW TRAINING ON FIRE "
                  f"{i + 1} / {len(self.loader.fire_lists)} ===============================")
            train, test = self.loader.load_fire(fire)

            #       Training
            # error todo: its still none with a real fire:
            if train is None or test is None:
                continue
            self.land_cover_grid, file_lcr, self.height_grid, self.fire_grid, self.weather_grid = train
            if (not np.any(self.fire_grid[:, :, 0])):
                print("Fire has no starting spot! skipping")
                continue

            # For first training iteration set land cover rates from file
            if land_cover_rates is None:
                land_cover_rates = file_lcr

            train_cost, new_land_cover_rates, new_params = self.optimize(params, land_cover_rates)
            params = new_params
            land_cover_rates = new_land_cover_rates

            #       Testing

            self.land_cover_grid, file_lcr, self.height_grid, self.fire_grid, self.weather_grid = test
            if self.land_cover_grid is None:
                continue
            shaped_params = params.transpose().reshape(len(params), 1)
            shaped_lcr = land_cover_rates.transpose().reshape(len(land_cover_rates), 1)
            test_cost = self.get_fitness(shaped_lcr, shaped_params)[0]
            training_costs.append(train_cost)
            test_costs.append(test_cost)
            print(f"TRAIN COST: {train_cost}")
            print(f"TEST COST: {test_cost}")

        avg_train_cost = np.array(training_costs)[np.not_equal(training_costs, -1)].mean()
        print(f"avg train cost {avg_train_cost}")
        avg_test_cost = np.array(test_costs)[np.not_equal(test_costs, -1)].mean()
        print(f"avg test cost {avg_test_cost}")
        np.save(f"test_results_single.npy", test_costs)
        np.save(f"train_results_single.npy", training_costs)

    def get_fitness(self, lcr, params):
        show_plots = True

        result = cuda_python.batch_simulate(self.land_cover_grid, lcr, self.height_grid,
                                            self.fire_grid, self.weather_grid,
                                            params)
        # Compare burnt area result with self.burnt_area_end
        burnt_area = self.fire_grid[:, :, self.fire_grid.shape[2] - 1] > 0.8
        if show_plots:
            first_burnt_area = self.fire_grid[:, :, 0]
        small_width = self.fire_grid.shape[0] // 1
        small_height = self.fire_grid.shape[1] // 1
        small_ba = cv2.resize(burnt_area.astype(np.int16), dsize=(small_height, small_width)) > 0.5
        flat_ba = small_ba.flatten()
        auc = np.zeros(result.shape[2])
        for i in range(result.shape[2]):
            small_sba = cv2.resize(result[:, :, i], dsize=(small_width, small_height)) < 0.5
            flat_sba = small_sba.flatten()
            auc[i] = metrics.roc_auc_score(flat_ba, flat_sba)

            if False and auc[i] > self.best_auc_for_plot:
                self.best_auc_for_plot = auc[i]
                fpr, tpr, threshold = metrics.roc_curve(flat_ba, flat_sba)

                plt.tight_layout()
                fig, axs = plt.subplots(2, 2)
                plt.subplots_adjust(wspace=.1, hspace=.4)

                axs[0, 0].imshow(first_burnt_area, interpolation='nearest')
                axs[0, 0].set_title("Starting point")

                axs[0, 1].plot(fpr, tpr, linestyle='--')
                axs[0, 1].set_title(f"AUC: {round(auc[i] * 100) / 100}")
                axs[0, 1].set_xlabel('False Positive Rate')
                axs[0, 1].set_ylabel('True Positive Rate')

                axs[1, 0].imshow(small_sba, interpolation='nearest')
                axs[1, 0].set_title(f"Simulated burnt area")

                axs[1, 1].imshow(small_ba, interpolation='nearest')
                axs[1, 1].set_title('Actual burnt area')

                plt.savefig(f"../new/imgs/auc_pic_{time.time()}_{auc[i]}.png", dpi=500)
                plt.show()

        return 1 - auc

    def optimize(self, initial_params, land_cover_rates):
        def wrap_self(inst):
            def f(x):
                lcr = x[:, 0:len(land_cover_rates)].transpose()
                params = x[:, len(land_cover_rates):].transpose()
                return inst.get_fitness(lcr, params)

            return f

        dimensions = land_cover_rates.shape[0] + len(initial_params)
        max_bound = 5 * np.ones(dimensions)
        min_bound = np.zeros(dimensions)
        bounds = (min_bound, max_bound)
        initial_values = np.zeros((self.swarm_size, dimensions))
        initial_values[:, 0:len(land_cover_rates)] = land_cover_rates
        initial_values[:, len(land_cover_rates):] = initial_params
        optimizer = ps.single.GlobalBestPSO(n_particles=self.swarm_size,
                                            init_pos=initial_values,
                                            dimensions=dimensions,
                                            options=self.options,
                                            bounds=bounds)

        # Perform optimization
        cost, pos = optimizer.optimize(wrap_self(self), iters=self.iterations)
        if False:
            plot_cost_history(cost_history=optimizer.cost_history)
        plt.show()
        return cost, pos[0:len(land_cover_rates)], pos[len(land_cover_rates):]


if __name__ == "__main__":
    pso = PSO()
    pso.full_train()
