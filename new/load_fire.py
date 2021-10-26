import os
import random

from data_paths import glob_fire_path, fire_meta_path
import geopandas
import numpy as np


# Polygon zit klinkklaar in die geometry
# kijk naar simulation code om te zien hoe ik deze polygons daar in implementeer
# Split deze data op in polygon groups?
# Kies 1 polygon group (met zelfde properties -> id) om mee te werken
# Ze start punt als start punt van de simulatie
# Vergelijk elke dag met de polygon van het vuur voor fitness en maak grafiek ervan voor data
# Test en train set!

# Hoe kies je een polygon om te gebruiken?
# automatisch training?

def pick_fire_id(bounds):
    if not os.path.isfile(fire_meta_path):
        generate_fire_meta()
    fire_list = np.load(fire_meta_path)
    random_index = random.randrange(fire_list.shape[0])
    print('loaded fire meta')
    # loop through fires
    # per fire get bounds of fire cropped gdf
    # check if bounds are full within restricted bounds parameter
    pass


def generate_fire_meta():
    index = 0
    batch_size = 10000
    current_id = -1
    id_start_offset = 0

    # all_ids: fire_id, row_start_index, row_end_index, bounds(left,right,bottom,top),
    all_ids = np.empty((0, 7), dtype=np.float)
    while True:
        gdf = geopandas.read_file(
            glob_fire_path,
            rows=slice(index, index + batch_size)
        )
        if gdf.empty:
            break

        sub_offset = 0
        for fire_id in gdf.values[:, 2]:
            sub_offset += 1
            if current_id == fire_id:
                continue
            if current_id != -1:
                id_end_offset = index + sub_offset
                bounds = gdf_to_bounds(geopandas.read_file(
                    glob_fire_path, rows=slice(id_start_offset, id_end_offset)
                ))
                new_id_row = np.array([
                    current_id, id_start_offset, id_end_offset,
                    bounds['left'], bounds['right'], bounds['bottom'], bounds['top']
                ])
                all_ids = np.vstack([all_ids, new_id_row])
            current_id = fire_id
            id_start_offset = index + sub_offset

        index += batch_size
        print(f"Scanning gdf... {index} rows scanned, {all_ids.shape[0]} fires found")

    print(f"Saving fire metadata to file {fire_meta_path}")
    np.save(fire_meta_path, all_ids)
    print(all_ids)


def load_fire_gdf(fire_id):
    print(os.path.abspath(glob_fire_path))

    index = 0
    batch_size = 10000
    fire_gdf = None
    while True:
        gdf = geopandas.read_file(
            glob_fire_path,
            rows=slice(index, index + batch_size)
        )
        partial_fire = gdf[gdf["Id"] == fire_id]
        if fire_gdf is None:
            fire_gdf = partial_fire
        else:
            fire_gdf.append(partial_fire)
        if partial_fire.shape[0] != gdf.shape[0]:
            break
        index += batch_size
    return fire_gdf


def gdf_to_bounds(gdf):
    if gdf.shape[0] == 1:
        return {
            "left": gdf.bounds['minx'].iloc[0],
            "right": gdf.bounds['maxx'].iloc[0],
            "bottom": gdf.bounds['miny'].iloc[0],
            "top": gdf.bounds['maxy'].iloc[0],
        }
    minx = min(*gdf.bounds['minx'])
    miny = min(*gdf.bounds['miny'])
    maxx = max(*gdf.bounds['maxx'])
    maxy = max(*gdf.bounds['maxy'])
    return {
        "left": minx,
        "right": maxx,
        "bottom": miny,
        "top": maxy,
    }


def bounds_to_square(bounds):
    left = bounds['left']
    right = bounds['right']
    top = bounds['top']
    bottom = bounds['bottom']
    width = right - left
    height = top - bottom
    center_y = bottom + (top - bottom) / 2
    center_x = left + (right - left) / 2
    if width > height:
        grid_bounds = {
            "left": left,
            "right": right,
            "bottom": center_y - width / 2,
            "top": center_y + width / 2,
        }
    else:
        grid_bounds = {
            "left": center_x - width / 2,
            "right": center_x + width / 2,
            "bottom": bottom,
            "top": top,
        }
    print("bounds is now a square around the area affected by the fire in `gdf`", grid_bounds)
    return grid_bounds


if __name__ == '__main__':
    pass
