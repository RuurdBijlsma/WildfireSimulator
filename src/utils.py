import pandas as pd
from shapely.geometry import Point
import geopandas
import matplotlib.pyplot as plt


def plot_bounds(bounds):
    width = bounds['right'] - bounds['left']
    height = bounds['top'] - bounds['bottom']
    world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'),
                                bbox=([
                                    bounds['left'] - width / 2,
                                    bounds['bottom'] - height / 2,
                                    bounds['right'] + width / 2,
                                    bounds['top'] + height / 2
                                ]))
    world.plot()
    x1, y1 = [bounds['left'], bounds['right']], [bounds['top'], bounds['top']]
    x2, y2 = [bounds['left'], bounds['right']], [bounds['bottom'], bounds['bottom']]
    x3, y3 = [bounds['left'], bounds['left']], [bounds['bottom'], bounds['top']]
    x4, y4 = [bounds['right'], bounds['right']], [bounds['bottom'], bounds['top']]
    # x1 = [-100, 0]
    # y1 = [50, 50]
    plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, color='r')
    plt.title('Calculated data boundary')
    plt.xlabel("Longitude (degrees)")
    plt.ylabel("Latitude (degrees)")
    plt.show()


def get_geo_bounds(top, left, bottom, right):
    df = pd.DataFrame({
        'Latitude': [bottom, top],  # Y value
        'Longitude': [left, right]  # X value
    })
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    frame = geopandas.GeoDataFrame(df, geometry='Coordinates')
    frame.set_crs(epsg=4326, inplace=True)
    return frame


def change_crs(x: float, y: float, from_crs=4326, to_crs=3857):
    df = pd.DataFrame({
        'Latitude': [y],  # Y value
        'Longitude': [x]  # X value
    })
    df['Coordinates'] = list(zip(df.Longitude, df.Latitude))
    df['Coordinates'] = df['Coordinates'].apply(Point)
    frame = geopandas.GeoDataFrame(df, geometry='Coordinates')
    frame.set_crs(epsg=from_crs, inplace=True)
    frame.to_crs(epsg=to_crs, inplace=True)
    point = frame['Coordinates'].iloc[0]
    return point.x, point.y
