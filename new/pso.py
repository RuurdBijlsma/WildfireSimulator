import numpy as np
import pyswarms as ps
from matplotlib import pyplot as plt
from load_all import FireLoader
from sklearn import metrics
import cv2
import time
import cuda_python
from statistics import mean


def save_results(train_results, test_results, fold=0):
    np.save(f"test_results_fold{fold}.npy", test_results.filled(-1))
    np.save(f"train_results_fold{fold}.npy", train_results.filled(-1))


class PSO:
    show_plots = False
    # PSO Options
    options = {'c1': 0.5, 'c2': 0.5, 'w': 0.9}
    swarm_size = 200
    iterations = 30
    # K-fold options
    data_size = 50
    learning_rate = .1
    n_folds = 5

    best_auc_for_plot = 0

    def __init__(self):
        self.loader = FireLoader(max_data=self.data_size, n_splits=self.n_folds)
        self.land_cover_grid = None
        self.height_grid = None
        self.fire_grid = None
        self.weather_grid = None

    def full_train(self):
        fold = 0
        start_fold = 4
        train_results = None
        test_results = None
        for train_indices, test_indices in self.loader.kf.split(self.loader.fire_lists):
            fold += 1
            if fold < start_fold:
                continue
            params = np.array([.3, .1, .1, 2, .1, 2, .2, 1, .1, 1])
            land_cover_rates = None
            training_costs = []
            train_mask = []
            print(f"FOLD {fold} / {self.n_folds} ===============================[ TRAINING ]==========================")
            for i, train_index in enumerate(train_indices):
                self.best_auc_for_plot = 0
                print(f"FOLD {fold} / {self.n_folds} =========================== NOW TRAINING ON FIRE "
                      f"{i + 1} / {len(train_indices)} ===============================")
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
            np.save(f'fold{fold}_params.npy', params)
            np.save(f'fold{fold}_land_cover_rates.npy', land_cover_rates)
            print(f"Saved params & land cover rates for fold {fold}")
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
            save_results(train_results, test_results, fold)
        k_fold_score = np.mean(test_results)
        print(f"Overal average training cost {np.mean(train_results)}")
        print(f"KFold score = {k_fold_score}")
        print("Test scores per fold: ", np.mean(test_results, axis=1))
        save_results(train_results, test_results)

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

                plt.savefig(f"imgs/auc_pic_{time.time()}_{auc[i]}.png", dpi=500)
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
