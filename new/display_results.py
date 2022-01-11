import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


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
    ax.set_title("GPU & CPU performance over grid size")
    ax.plot(sizes, gpu_averages, label="GPU")
    ax.plot(sizes, cpu_averages, label="CPU")
    ax.set_xlabel('Grid size (NxN)')
    ax.set_ylabel('Time (μs)')
    ax.legend()
    fig.show()
    fig.savefig("gpu_vs_cpu.png", dpi=1000)


def plot_gpu():
    gpu_averages = [105299.2, 94031.2, 83607.2, 101546.6, 98102.8, 86909.8, 89334.4, 95131.8, 98602.4]
    sizes = [2, 10, 20, 40, 60, 80, 100, 120, 140]
    fig, ax = plt.subplots()
    ax.set_title("GPU performance over grid size")
    ax.plot(sizes, gpu_averages, label="GPU")
    ax.set_xlabel('Grid size (NxN)')
    ax.set_ylabel('Time (μs)')
    fig.show()


plot_results()
