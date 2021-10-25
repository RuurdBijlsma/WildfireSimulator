import os
from data_paths import glob_fire_path
import geopandas


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
    id = geopandas.read_file(glob_fire_path, rows=slice(0, 1)).iloc[0]["Id"]
    # loop through fires
    # per fire get bounds of fire cropped gdf
    # check if bounds are full within restricted bounds parameter
    return id


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
    minx = min(*gdf.bounds['minx'])
    miny = min(*gdf.bounds['miny'])
    maxx = max(*gdf.bounds['maxx'])
    maxy = max(*gdf.bounds['maxy'])
    width = maxx - minx
    height = maxy - miny
    center_y = miny + (maxy - miny) / 2
    center_x = minx + (maxx - minx) / 2
    if width > height:
        grid_bounds = {
            "left": minx,
            "right": maxx,
            "bottom": center_y - width / 2,
            "top": center_y + width / 2,
        }
    else:
        grid_bounds = {
            "left": center_x - width / 2,
            "right": center_x + width / 2,
            "bottom": miny,
            "top": maxy,
        }
    print("bounds is now a square around the area affected by the fire in `gdf`", grid_bounds)
    return grid_bounds


if __name__ == '__main__':
    pass
