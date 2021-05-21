import geopandas
import os
import netCDF4 as nc


def run():
    ff = os.path.abspath("data/burnt/c_gls_BA300_201607200000_GLOBE_PROBAV_V1.0.1.nc")
    ds = nc.Dataset(ff)
    print(ds)


if __name__ == '__main__':
    run()
