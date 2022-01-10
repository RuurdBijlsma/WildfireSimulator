import netCDF4 as nc
import os
import numpy as np
from matplotlib import pyplot as plt
from regions import region
import geopandas
from shapely.geometry import Point
from constants import height_path


def load_height(bounds):
    ds = nc.Dataset(height_path)
    # print(ds)
    layer = "ASTER_GDEM_DEM"

    lats = ds['lat'][:].filled(0)
    lons = ds['lon'][:].filled(0)
    i_left = np.argmax(lons >= bounds['left'])
    i_right = np.argmax(lons >= bounds['right'])
    i_bottom = np.argmax(lats <= bounds['bottom'])
    i_top = np.argmax(lats <= bounds['top'])
    heights = ds[layer][:, i_top:i_bottom, i_left:i_right].filled(0).squeeze()

    plot_height_data = False
    if plot_height_data:
        # height_img = np.squeeze(ds[layer][:, :, :])
        plt.imshow(heights, interpolation='nearest')
        title = os.path.basename(height_path)
        plt.title(title)
        plt.show()

    # print(i_left)
    return heights


if __name__ == '__main__':
    df = region["bounds"]
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    frame = geopandas.GeoDataFrame(df, geometry='Coordinates')
    frame.set_crs(epsg=4326, inplace=True)
    [left, bottom, right, top] = frame.total_bounds
    load_height(1)
