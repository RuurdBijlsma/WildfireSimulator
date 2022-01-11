import numpy as np
import pyswarms as ps
from matplotlib import pyplot as plt
from load_all import FireLoader
from sklearn import metrics
import cuda_python
from statistics import mean


class PSO:
    show_plots = False
    # PSO Options
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
    swarm_size = 30
    iterations = 30
    # K-fold options
    data_size = 30
    learning_rate = .2
    n_folds = 5

    def __init__(self):
        self.loader = FireLoader(max_data=self.data_size, n_splits=self.n_folds)
        self.land_cover_grid = None
        self.height_grid = None
        self.fire_grid = None
        self.weather_grid = None

    def full_train(self):
        fold = 1
        train_results = None
        test_results = None
        for train_indices, test_indices in self.loader.kf.split(self.loader.fire_lists):
            params = np.array([.3, .1, .1, 2, .1, 2, .2, 1, .1, 1])
            land_cover_rates = None
            training_costs = []
            train_mask = []
            print(f"FOLD {fold} / {self.n_folds} ===============================[ TRAINING ]==========================")
            for i, train_index in enumerate(train_indices):
                print(f"FOLD {fold} / {self.n_folds} =========================== NOW TRAINING ON FIRE "
                      f"{i + 1} / {len(train_indices)} ===============================")
                # if i < 163:
                #     continue
                self.land_cover_grid, file_lcr, self.height_grid, self.fire_grid, self.weather_grid \
                    = self.loader.load_fire(self.loader.fire_lists[train_index])
                if self.land_cover_grid is None:
                    training_costs.append(-1)
                    train_mask.append(1)
                    continue
                # For first training iteration set land cover rates from file
                if land_cover_rates is None:
                    land_cover_rates = file_lcr

                cost, new_land_cover_rates, new_params = self.optimize(params, land_cover_rates)
                params = params + (new_params - params) * self.learning_rate
                land_cover_rates = land_cover_rates + (new_land_cover_rates - land_cover_rates) * self.learning_rate
                train_mask.append(0)
                training_costs.append(cost)

            avg_train_cost = np.ma.array(training_costs, mask=train_mask).mean()
            print(f"FOLD {fold} / {self.n_folds} Avg training cost {avg_train_cost}")
            test_costs = []
            test_mask = []
            print(f"FOLD {fold} / {self.n_folds} ===========================[ TESTING ]===============================")
            for i, test_index in enumerate(test_indices):
                print(f"FOLD {fold} / {self.n_folds} Testing {i + 1} / {len(test_indices)}")
                self.land_cover_grid, file_lcr, self.height_grid, self.fire_grid, self.weather_grid \
                    = self.loader.load_fire(self.loader.fire_lists[test_index])
                if land_cover_rates is None:
                    land_cover_rates = file_lcr
                if self.land_cover_grid is None:
                    test_costs.append(-1)
                    test_mask.append(1)
                    continue
                shaped_params = params.transpose().reshape(len(params), 1)
                shaped_lcr = land_cover_rates.transpose().reshape(len(land_cover_rates), 1)
                cost = self.get_fitness(shaped_lcr, shaped_params)[0]
                test_costs.append(cost)
                test_mask.append(0)
            fold += 1

            test_ma = np.ma.array(test_costs, mask=test_mask)
            print(f"FOLD {fold} / {self.n_folds} Avg test cost {test_ma.mean()}")
            train_ma = np.ma.array(training_costs, mask=train_mask)
            if train_results is None or test_results is None:
                train_results = train_ma
                test_results = test_ma
            else:
                train_results = np.ma.vstack([train_results, train_ma])
                test_results = np.ma.vstack([test_results, test_ma])
                print(5)
        k_fold_score = np.mean(test_results)
        print(f"Overal average training cost {np.mean(train_results)}")
        print(f"KFold score = {k_fold_score}")
        print("Test scores per fold: ", np.mean(test_results, axis=1))
        np.save("test_results.npy", test_results.filled(-1))
        np.save("train_results.npy", train_results.filled(-1))

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
        # plotters.plot_cost_history(cost_history=optimizer.cost_history)
        # plt.show()
        return cost, pos[0:len(land_cover_rates)], pos[len(land_cover_rates):]


if __name__ == "__main__":
    pso = PSO()
    pso.full_train()
