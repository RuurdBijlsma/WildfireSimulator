import os

glob_fire_dir = "../data/GlobFire3/Full_GlobFireV2_Jan_2021/"
height_path = "../data/Height/ASTGTM_NC.003_30m_aid0001.nc"
weather_path = "../data/Weather/adaptor.mars.internal-1630357570.1783972-1285-14-bb21d314-73d9-4199-a02a-c883a678b6de.nc"


def get_fire_meta_path(x):
    return f'../data/fire_meta_{os.path.basename(x)}.npy'
