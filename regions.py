import os

import pandas as pd

cyprus = {
    "bounds": pd.DataFrame({
        'Latitude': [34.8, 35.1],  # Y value
        'Longitude': [32.9, 33.3]  # X value
    }),
    "nc_paths": [
        os.path.abspath("data/burnt/c_gls_BA300_201609100000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201609200000_GLOBE_PROBAV_V1.0.1.nc"),
    ],
}

# lonlat top left       40.00852,37.39260
# lonlat bottom right   40.79075,36.92898
south_east_turkey = {
    "bounds": pd.DataFrame({
        'Latitude': [36.92898, 37.39260],  # Y value
        'Longitude': [40.00852, 40.79075]  # X value
    }),
    "nc_paths": [
        os.path.abspath("data/burnt/c_gls_BA300_201606300000_GLOBE_PROBAV_V1.0.1.nc"),
        os.path.abspath("data/burnt/c_gls_BA300_201607100000_GLOBE_PROBAV_V1.0.1.nc"),
    ],
}

region = south_east_turkey
