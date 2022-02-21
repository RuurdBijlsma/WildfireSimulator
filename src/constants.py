import os

glob_fire_dir = "../data/GlobFire3/Full_GlobFireV2_Jan_2021/"
height_path = "../data/Height/southeurope/ASTGTM_NC.003_30m_aid0001.nc"
weather_path = "../data/Weather/southeurope/adaptor.mars.internal-1641762410.3048615-28640-10-e67acac7-510a-4b87-8cfc-a6e2c6f99fb4.nc"
land_cover_path = "../data/DATA/U2018_CLC2018_V2020_20u1.gpkg"
seed = 80


def get_fire_meta_path(x):
    return f'../data/fire_meta_{os.path.basename(x)}.npy'
