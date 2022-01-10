import geopandas
import os
import matplotlib.pyplot as plt
from constants import land_cover_path
from utils import get_geo_bounds
import matplotlib.patches as patches


def load_land_cover(bounds):
    # left, top = change_crs(bounds['left'], bounds['top'], from_crs=4326, to_crs=3857)
    # right, bottom = change_crs(bounds['right'], bounds['bottom'], from_crs=4326, to_crs=3857)
    geo_bounds = get_geo_bounds(bounds['top'], bounds['left'], bounds['bottom'], bounds['right'])

    p = os.path.abspath(land_cover_path)
    gdf = geopandas.read_file(p, bbox=geo_bounds)
    gdf.to_crs(epsg=4326, inplace=True)
    plot_land_cover = False
    if plot_land_cover:
        rect = patches.Rectangle((bounds['left'], bounds['bottom']),
                                 (bounds['right'] - bounds['left']),
                                 (bounds['top'] - bounds['bottom']),
                                 linewidth=1, edgecolor='r', facecolor='none')
        fig, ax = plt.subplots()
        gdf.plot("Code_18", ax=ax)
        ax.add_patch(rect)
        plt.show()
    return gdf
