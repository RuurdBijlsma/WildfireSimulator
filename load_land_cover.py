import geopandas
import os
import matplotlib.pyplot as plt


def run():
    p = os.path.abspath("data/DATA/U2018_CLC2018_V2020_20u1.gpkg")
    print(p)
    gdf_mask = geopandas.read_file(geopandas.datasets.get_path("naturalearth_lowres"))
    # gdf = geopandas.read_file(
    #     p,
    #     mask=gdf_mask[gdf_mask.name == "Cyprus"],
    # )
    #                                   B L (x)  B L (y)  B R (x)   T R (y)
    # gdf = geopandas.read_file(p, bbox=([6350000, 1600000, 6500000,  1700000]))
    gdf = geopandas.read_file(p, bbox=([6425000, 1625000, 6430000,  1650000]))
    # gdf = geopandas.read_file(p, bbox=(1560952.51, 942165.13, 2033490.33, 1541354.67))
    # gdf = geopandas.read_file(p)
    print(gdf.bounds)
    print(gdf.head())
    gdf.plot("Code_18")
    plt.show()
    print("DONE")


if __name__ == '__main__':
    run()
