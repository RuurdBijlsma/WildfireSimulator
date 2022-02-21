import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def plot_params():
    params_per_fold = [
        np.load("../results/fold1_params.npy"),
        np.load("../results/fold2_params.npy"),
        np.load("../results/fold3_params.npy"),
        np.load("../results/fold4_params.npy"),
        np.load("../results/fold5_params.npy"),
    ]
    params_labels = ["burnRate", "heightEffectMultiplierUp", "heightEffectMultiplierDown", "windEffectMultiplier",
                     "activityThreshold", "spreadSpeed", "deathRate", "areaEffectMultiplier", "fireDeathThreshold",
                     "cellArea"]
    plot_param_values(params_per_fold, params_labels, "param")

    lcr_per_fold = [
        np.load("../results/fold1_land_cover_rates.npy"),
        np.load("../results/fold2_land_cover_rates.npy"),
        np.load("../results/fold3_land_cover_rates.npy"),
        np.load("../results/fold4_land_cover_rates.npy"),
        np.load("../results/fold5_land_cover_rates.npy"),
    ]
    lcr_labels = ["Continuous urban fabric", "Discontinuous urban fabric", "Industrial or commercial units",
                  "Road and rail networks", "Port areas", "Airports", "Mineral extraction sites",
                  "Dump sites", "Construction sites", "Green urban areas", "Sport and leisure facilities",
                  "Non-irrigated arable land", "Permanently irrigated land", "Rice fields", "Vineyards",
                  "Fruit trees and berry plantations", "Olive groves", "Pastures",
                  "Annual crops", "Complex cultivation patterns",
                  "Agriculture",
                  "Agro-forestry areas", "Broad-leaved forest", "Coniferous forest", "Mixed forest",
                  "Natural grasslands", "Moors and heathland", "Sclerophyllous vegetation",
                  "Transitional woodland-shrub", "Beaches, dunes, sands", "Bare rocks", "Sparsely vegetated areas",
                  "Burnt areas", "Glaciers and perpetual snow", "Inland marshes", "Peat bogs", "Salt marshes",
                  "Salines", "Intertidal flats", "Water courses", "Water bodies", "Coastal lagoons", "Estuaries",
                  "Sea and ocean", "NODATA", "UNCLASSIFIED LAND SURFACE", "UNCLASSIFIED WATER BODIES"]
    plot_param_values(lcr_per_fold, lcr_labels, "lcr", True)


def plot_param_values(values, labels, name, wide=False):
    for i, data in enumerate(values):
        plt.scatter(labels, data, label=f"Fold {i + 1}")
    fig = plt.gcf()
    fig.set_size_inches(15 if wide else 5, 7 if wide else 5)
    fig.set_dpi(200)
    plt.xlabel("Parameter")
    plt.ylabel("Optimized value")
    plt.legend()
    plt.xticks(rotation=60, ha='right')
    plt.tight_layout(h_pad=1)
    plt.savefig(f"param_plot_{name}.png", dpi=300)
    plt.show()


def plot_cuda():
    times = [5942, 4828, 4311, 3912, 3732, 4321, 3932, 3981, 3946, 3489, 3992, 3591, 3210, 3456, 3473, 3475, 3429, 3544,
             3993, 3850, 3500, 4090, 3905, 3866, 3827, 3999, 4291, 4123, 4222, 4536, 4378, 4314, 4284, 4526, 4838, 4754,
             4770, 4764, 4717, 4728, 4870, 4756]
    threads = [12, 24, 36, 48, 60, 72, 84, 96, 108, 120, 132, 144, 156, 168, 180, 192, 204, 216, 228, 240, 252, 264,
               276, 288, 300, 312, 324, 336, 348, 360, 372, 384, 396, 408, 420, 432, 444, 456, 468, 480, 492, 504]
    plt.plot(threads, times)
    plt.xlabel("GPU Threads")
    plt.ylabel("Time (μs)")
    plt.title("Speed over GPU thread count")
    plt.savefig("gpu_threads.png", dpi=500)
    plt.show()


def plot_results():
    test_data = np.load('results/best/test_results.npy')
    test_mask = (test_data == -1) * 1
    train_data = np.load('results/best/train_results.npy')
    train_mask = (train_data == -1) * 1

    train = np.ma.array(train_data, mask=train_mask)
    test = np.ma.array(test_data, mask=test_mask)

    print(f"Kfold score = {test.mean()}")
    print("Test result per fold")
    print(test.mean(axis=1))

    fig, ax = plt.subplots()
    ax.plot(test.transpose())
    fig.show()
    ax.plot(train.transpose())
    fig.show()
    print(5)


def plot_gpu_vs_cpu():
    gpu_averages = [105299.2, 94031.2, 83607.2, 101546.6, 98102.8, 86909.8, 89334.4, 95131.8, 98602.4]
    cpu_averages = [117767.6, 263677, 727888.4, 2360881.2, 5244902.6, 9325492.4, 14421761.6, 20978658, 28291146]
    sizes = [2, 10, 20, 40, 60, 80, 100, 120, 140]
    fig, ax = plt.subplots()
    ax.set_title("GPU & CPU speed over grid size")
    ax.plot(sizes, gpu_averages, label="GPU")
    ax.plot(sizes, cpu_averages, label="CPU")
    ax.set_xlabel('Grid size (NxN)')
    ax.set_ylabel('Time (μs)')
    ax.legend()
    fig.show()
    fig.savefig("gpu_vs_cpu.png", dpi=500)


def plot_gpu():
    gpu_averages = [105299.2, 94031.2, 83607.2, 101546.6, 98102.8, 86909.8, 89334.4, 95131.8, 98602.4]
    sizes = [2, 10, 20, 40, 60, 80, 100, 120, 140]
    fig, ax = plt.subplots()
    ax.set_title("GPU speed over grid size")
    ax.plot(sizes, gpu_averages, label="GPU")
    ax.set_xlabel('Grid size (NxN)')
    ax.set_ylabel('Time (μs)')
    fig.show()


# plot_results()
# plot_cuda()
plot_params()
