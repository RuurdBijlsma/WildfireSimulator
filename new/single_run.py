import numpy as np
from load_all import FireLoader
import cuda_python
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def main():
    loader = FireLoader(max_data=500, n_splits=4)
    # for i in range(317):
    #     if i < 38:
    #         continue
    i = 38
    land_cover_grid, file_lcr, height_grid, fire_grid, weather_grid = loader.load_fire(loader.fire_lists[i])
    if land_cover_grid is None:
        return
    print(f"{i}: FIRE DIM", fire_grid.shape)

    lcr_grid = np.zeros((fire_grid.shape[0], fire_grid.shape[1]))
    for x in range(lcr_grid.shape[0]):
        for y in range(lcr_grid.shape[1]):
            lcr_grid[x, y] = file_lcr[land_cover_grid[x, y]]

    fig, axs = plt.subplots(2, 2)
    plt.subplots_adjust(wspace=.1, hspace=.4)

    axs[0, 0].imshow(lcr_grid, interpolation='nearest')
    axs[0, 0].set_title("Land cover grid")

    axs[0, 1].imshow(height_grid, interpolation='nearest')
    axs[0, 1].set_title("Elevation")

    axs[1, 0].imshow(weather_grid[:, :, 0, 0], interpolation='nearest')
    axs[1, 0].set_title("Wind east")

    axs[1, 1].imshow(weather_grid[:, :, 0, 1], interpolation='nearest')
    axs[1, 1].set_title("Wind north")

    plt.savefig(f"imgs/data_pic{i}.png", dpi=500)
    plt.show()

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(fire_grid[:, :, 0], interpolation='nearest')
    axs[0].set_title("Initial fire")
    axs[1].imshow(fire_grid[:, :, fire_grid.shape[2] - 1], interpolation='nearest')
    axs[1].set_title("Final fire")
    plt.savefig(f"imgs/fires_{i}.png", dpi=500)
    plt.show()

    params = np.array([.3, .1, .1, 2, .1, 2, .2, 1, .1, 1])
    lcr = file_lcr.reshape((file_lcr.shape[0], 1))
    params = params.reshape((params.shape[0], 1))
    result = cuda_python.batch_simulate(land_cover_grid, lcr, height_grid, fire_grid[:, :, 0], weather_grid, params)


if __name__ == "__main__":
    main()
