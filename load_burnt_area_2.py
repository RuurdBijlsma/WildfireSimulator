import fiona

shape = fiona.open("data/GlobFire3/Full_GlobFireV2_Jan_2021/MODIS_BA_GLOBAL_1_1_2020.shp")
print(shape.schema)
# {'geometry': 'LineString', 'properties': OrderedDict([(u'FID', 'float:11')])}
# first feature of the shapefile
first = shape.next()
print(first)  # (GeoJSON format)
