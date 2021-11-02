import os
import random

from data_paths import get_fire_meta_path
import geopandas
import numpy as np


# return fire id of randomly picked fire within bounds
def pick_fire(bounds):
    fire_paths = random.sample(bounds['fire_paths'], len(bounds['fire_paths']))
    for fire_path in fire_paths:
        fire_meta_path = get_fire_meta_path(fire_path)
        if not os.path.isfile(fire_meta_path):
            generate_fire_meta(fire_path, fire_meta_path)
        fire_list = np.load(fire_meta_path)
        in_bounds = np.all([
            fire_list[:, 3] >= bounds['left'],
            fire_list[:, 4] <= bounds['right'],
            fire_list[:, 5] >= bounds['bottom'],
            fire_list[:, 6] <= bounds['top'],
        ], axis=0)
        filtered_fires = fire_list[in_bounds]
        if filtered_fires.shape[0] == 0:
            continue
        random_index = random.randrange(filtered_fires.shape[0])
        fire_array = filtered_fires[random_index]
        return {
            'fire_path': fire_path,
            'id': fire_array[0],
            'row_start': fire_array[1],
            'row_end': fire_array[2],
            'bounds': {
                'left': fire_array[3],
                'right': fire_array[4],
                'bottom': fire_array[5],
                'top': fire_array[6],
            },
        }
    raise Exception("No fire found within bounds restrictions")


def generate_fire_meta(glob_fire_path, fire_meta_path):
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


def load_fire_gdf(fire):
    return geopandas.read_file(
        fire['fire_path'],
        rows=slice(fire['start_index'], fire['end_index'])
    )


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
