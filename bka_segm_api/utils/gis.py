#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import shutil
import numpy as np
import pandas as pd
import geopandas as gp
from osgeo import gdal, ogr, osr
import argparse
import logging
import matplotlib.pyplot as plt
from .core import add_ref_to_img


def polygonize_pmap(pmap_ds: gdal.Dataset, pmap_threshold=131, layer_name='unknown.gpkg', path_out=None) -> gdal.Dataset:
    pmap_bin_img = (pmap_ds.ReadAsArray() > pmap_threshold).astype(np.uint8)
    pmap_bin_ds = add_ref_to_img(pmap_bin_img, pmap_ds)
    #
    srs = osr.SpatialReference()
    srs.ImportFromWkt(pmap_ds.GetProjectionRef())
    #
    if path_out is None:
        ds_out = ogr.GetDriverByName('MEMORY').CreateDataSource('wrk')
        # ds_out = ogr.GetDriverByName('ESRI Shapefile').CreateDataSource(f'/vsimem/{layer_name}.shp')
    else:
        drv = ogr.GetDriverByName('GPKG')
        if os.path.isfile(path_out):
            drv.DeleteDataSource(path_out)
        ds_out = drv.CreateDataSource(path_out)
        layer_name = os.path.splitext(os.path.basename(path_out))[0]
    ds_layer = ds_out.CreateLayer(layer_name, geom_type=ogr.wkbPolygon, srs=srs)
    fd = ogr.FieldDefn('DN', ogr.OFTInteger)
    ds_layer.CreateField(fd)
    dst_field = 0
    # path_pmap = '/home/ar/data/uiip/quarry_data_test/s2_u8_t3_msk.tif'
    ds_band = pmap_bin_ds.GetRasterBand(1)
    gdal.Polygonize(ds_band, ds_band, ds_layer, dst_field, [], callback=gdal.TermProgress)
    return ds_out


if __name__ == '__main__':
    pass