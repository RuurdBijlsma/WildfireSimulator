import math
from datetime import timedelta

class Grid:
    def __init__(self, bounds):
        # 1 grid cell is how big?: (coordinates)
        self.spatial_resolution = .002
        # 1 grid cell is how long?: (hours)
        self.temporal_resolution = timedelta(hours=1)

        coordinate_width = bounds['right'] - bounds['left']
        coordinate_height = bounds['right'] - bounds['left']
        self.timedelta = bounds['time_end'] - bounds['time_start']
        self.width = math.ceil(coordinate_width / self.spatial_resolution)
        self.height = math.ceil(coordinate_height / self.spatial_resolution)
        self.duration = self.timedelta / self.temporal_resolution

        print(self)
