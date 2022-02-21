C++ part of the code is located here: https://github.com/RuurdBijlsma/cuda-fire-sim

The compiled cuda_python.so file is not guaranteed to work on every os, tested on Ubuntu 20.04. Compile it yourself for
other operating systems. The packages required for this project are listed in `requirements.in`.

## Data

Download links for the required data:

* Corine Land Cover: https://land.copernicus.eu/pan-european/corine-land-cover/clc2018?tab=metadata
* GlobFire: https://gwis.jrc.ec.europa.eu/apps/country.profile/downloads
* ERA5-Land (weather): https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-land?tab=form
* Elevation: https://lpdaac.usgs.gov/products/astgtmv003/

## Installing packages

`pip install -r requirements.txt`

## Compiling `requirements.txt`

1. Make sure `pip-tools` is installed: `python -m pip install pip-tools`
2. `pip-compile requirements.in`