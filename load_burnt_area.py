import os
import netCDF4 as nc
import geopandas
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import Point
from regions import region


def show_nc(nc_path):
    df = region["bounds"]
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    frame = geopandas.GeoDataFrame(df, geometry='Coordinates')
    frame.set_crs(epsg=4326, inplace=True)
    [left, bottom, right, top] = frame.total_bounds

    ds = nc.Dataset(nc_path)
    layer = 'FDOB_DEKAD'
    # x = 71600
    # y = 15150  # lower is higher in the map
    # size = 400
    # bottom = y - size // 2
    # top = y + size // 2
    # left = x - size // 2
    # right = x + size // 2
    lats = ds['lat'][:].filled(0)
    lons = ds['lon'][:].filled(0)
    i_left = np.argmax(lons >= left)
    i_right = np.argmax(lons >= right)
    i_bottom = np.argmax(lats <= bottom)
    i_top = np.argmax(lats <= top)
    burns = ds[layer][i_top:i_bottom, i_left:i_right].filled(0)

    plt.imshow(burns, interpolation='nearest')
    title = os.path.basename(nc_path)
    plt.title(title)
    plt.show()


def run():
    ffs = [
        os.path.abspath("data/burnt/c_gls_BA300_201605100000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201605200000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201605310000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201606100000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201606200000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201606300000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201607100000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201607200000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201607310000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201608100000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201608200000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201608310000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201609100000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201609200000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201609300000_GLOBE_PROBAV_V1.0.1.nc"),
    ]
    for ff in ffs:
        show_nc(ff)
    # show_nc(ffs[0])


if __name__ == '__main__':
    run()
