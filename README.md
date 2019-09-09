# Quarry segmentation searcher (sentinel2)

Simple search for quarry areas by sentinel2
images (only inference part)

### Project dependencies:
 - [Keras](keras.io)/[Tensorflow-1.x](www.tensorflow.org) as DL framework 
 - [GDAL](gdal.org) as core geo-library
 - [GeoPandas](geopandas.org)/[Fiona](fiona.readthedocs.io) as lite-abstarction library for geometry
 - [Python](python.org) >= 3.7 as glue :) 
 - numpy/scipy/scikit-image/scikit-learn as DS/CV python stack
 - [FELS](https://github.com/vascobnunes/fetchLandsatSentinelFromGoogleCloud): simple CLI for google-cloud sentinel2 downloading
 - ... and much more

To avoid hell of library dependencies and
application deployment and launch use the
[Docker-file](docker/readme_docker.md) in project.

