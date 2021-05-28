import geopandas
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.basemap import Basemap


def run():
    ff = os.path.abspath("data/burnt/c_gls_BA300_201607200000_GLOBE_PROBAV_V1.0.1.nc")
    ds = nc.Dataset(ff)
    layer = 'FDOB_DEKAD'
    x = 71600
    y = 15150  # lower is higher in the map
    size = 200
    bottom = y - size
    top = y + size
    left = x - size
    right = x + size
    lats = ds['lat'][bottom:top]
    lons = ds['lon'][left:right]
    burns = ds[layer][bottom:top, left:right]
    print(burns)
    burns_units = ds.variables[layer].units
    ds.close()

    lon_0 = lons.mean()
    lat_0 = lats.mean()

    print(lat_0, lon_0)

    m = Basemap(width=100000, height=80000,
                resolution='i', projection='stere',
                lat_ts=40, lat_0=lat_0, lon_0=lon_0)

    # Because our lon and lat variables are 1D,
    # use meshgrid to create 2D arrays
    # Not necessary if coordinates are already in 2D arrays.
    lon, lat = np.meshgrid(lons, lats)
    xi, yi = m(lon, lat)

    # Plot Data
    cs = m.pcolor(xi, yi, np.squeeze(burns))

    # Add Grid Lines
    m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
    m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

    # Add Coastlines, States, and Country Boundaries
    m.drawcoastlines()
    m.drawstates()
    m.drawcountries()

    # Add Colorbar
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    cbar.set_label(burns_units)

    # Add Title
    plt.title(layer)

    plt.show()


if __name__ == '__main__':
    run()
