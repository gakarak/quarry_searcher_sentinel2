#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
from typing import Union as U, Optional as O
import numpy as np
import pandas as pd
import geopandas as gp
from osgeo import gdal, ogr, osr



def generate_tmp_path(path, root=None, tmp='.tmp'):
    if root is None:
        root = os.path.dirname(path)
    rel_ = os.path.relpath(path, root)
    ret = os.path.join(root, tmp, rel_)
    os.makedirs(os.path.dirname(ret), exist_ok=True)
    return ret


def add_ref_to_img(imgu8: np.ndarray, ds_ref, set_zero_nodata=True) -> gdal.Dataset:
    if isinstance(ds_ref, str):
        ds_ref = gdal.OpenEx(ds_ref)
    if imgu8.ndim > 2:
        sx, sy = imgu8.shape[1:][::-1]
        nch = imgu8.shape[0]
    else:
        sx, sy = imgu8.shape[::-1]
        nch = 1
    ds_out = gdal.GetDriverByName('Mem').Create('', sx, sy, nch, gdal.GDT_Byte)
    #
    srs = osr.SpatialReference()
    sx_ref = ds_ref.RasterXSize
    sy_ref = ds_ref.RasterYSize
    trf_ = list(ds_ref.GetGeoTransform())
    trf_[1] *= sx_ref / sx
    trf_[5] *= sy_ref / sy
    trf_ = tuple(trf_)
    srs.ImportFromWkt(ds_ref.GetProjection())
    ds_out.SetProjection(srs.ExportToWkt())
    ds_out.SetGeoTransform(trf_)
    if imgu8.ndim < 3:
        imgu8 = [imgu8]
    for bi, b in enumerate(imgu8):
        ds_band = ds_out.GetRasterBand(bi + 1)
        if set_zero_nodata:
            ds_band.SetNoDataValue(0)
        ds_band.WriteArray(b)
        ds_band.FlushCache()
    ds_out.FlushCache()
    return ds_out


def get_ds_resolution(ds: U[str, gdal.Dataset]) -> float:
    if isinstance(ds, str):
        ds = gdal.OpenEx(ds, 0)
    trf = ds.GetGeoTransform()
    rx, ry = trf[1], trf[5]
    return float(np.mean([abs(rx), abs(ry)]))


def rasterize_geom_mask(geom_xy: gp.GeoDataFrame, bbox_xy: tuple, res: int) -> gdal.Dataset:
    [xmin, ymin], [xmax, ymax] = bbox_xy
    sx, sy = int(round((xmax - xmin) / res)), int(round((ymax - ymin) / res))
    ds_out = gdal.GetDriverByName('Mem').Create('', sx, sy, 1, gdal.GDT_Byte)
    crs_xy = geom_xy.crs
    crs_bl = {'init': 'epsg:4326'}
    geom_bl = geom_xy.to_crs(crs_bl)
    # prj_xy = osr.SpatialReference()
    # prj_xy.ImportFromWkt(crs_xy)
    # ds_out.SetProjection(prj_xy.ExportToWkt())
    trf = [xmin, res, 0, ymax, 0, -res]
    ds_out.SetGeoTransform(trf)
    srs_prj = osr.SpatialReference()
    srs_prj.ImportFromEPSG(crs_xy.to_epsg())
    ds_out.SetProjection(srs_prj.ExportToWkt())
    #
    # ds_geom = gdal.OpenEx(geom_bl.to_json())
    ds_geom = ogr.Open(geom_bl.to_json()) #FIXME: This is Kostil fot GDAL < 2.3.X
    gdal.RasterizeLayer(ds_out, [1], ds_geom.GetLayer(), burn_values=[1], options=['ALL_TOUCHED=TRUE'])
    return ds_out


def __reproject_to_img(geom: gp.GeoDataFrame, path_img) -> gp.GeoDataFrame:
    ds_ = gdal.OpenEx(path_img)
    geom = geom.to_crs(ds_.GetProjection())
    return geom


def __get_geom_bbox(geom: gp.GeoDataFrame, pad: int = None) -> tuple:
    x1 = min(geom.bounds['minx'])
    x2 = max(geom.bounds['maxx'])
    y1 = min(geom.bounds['miny'])
    y2 = max(geom.bounds['maxy'])
    if pad is not None:
        x1 -= pad
        y1 -= pad
        x2 += pad
        y2 += pad
    bbox = ((x1, y1), (x2, y2))
    return bbox


def get_geom_bboxes(geom: gp.GeoDataFrame, pad: int, idx_def: str, attr: str = None, no_split=False) -> list:
    if no_split:
        bbox = __get_geom_bbox(geom, pad)
        ret = [{
            'geom': geom,
            'bbox': bbox,
            'idx':  idx_def
        }]
    else:
        ret = []
        if attr is None:
            num = len(geom)
            for ii in range(num):
                g = geom.iloc[ii:ii+1]
                ret.append({
                    'geom': g,
                    'bbox': __get_geom_bbox(g, pad),
                    'idx':  f'{ii:06d}'
                })
        else:
            for gi, g in geom.groupby(by=attr):
                ret.append({
                    'geom': g,
                    'bbox': __get_geom_bbox(g, pad),
                    'idx': gi
                })
    return ret


def pad_img_to_cropsize(img: np.ndarray, crop_size:int, pad_type='reflect') -> np.ndarray:
    nr, nc = img.shape[:2]
    new_size = crop_size * np.ceil(np.array(img.shape[:2]) / crop_size).astype(np.int)
    if img.ndim > 2:
        new_shape = (new_size[0], new_size[1], img.shape[2])
    else:
        new_shape = tuple(new_size)
    if img.ndim > 2:
        ret_ = np.pad(img, ((0, new_shape[0] - nr), (0, new_shape[1] - nc), (0, 0)), pad_type)
    else:
        ret_ = np.pad(img, ((0, new_shape[0] - nr), (0, new_shape[1] - nc)), pad_type)
    return ret_


def read_img_into_memory(path_img: str, ret_batch=True, norm_coef=255., pad_size=None, remove_last_channel=True) -> np.ndarray:
    img_batch = gdal.OpenEx(path_img).ReadAsArray().transpose((1, 2, 0))
    if remove_last_channel:
        img_batch = img_batch[..., :-1]
    if norm_coef is not None:
        img_batch = img_batch.astype(np.float32) / norm_coef
    if pad_size is not None:
        img_batch = pad_img_to_cropsize(img_batch, crop_size=pad_size)
    if ret_batch:
        img_batch = img_batch[None, ...].copy()
    return img_batch


def map_u16_to_u8(im: np.ndarray, diap: tuple) -> np.ndarray:
    d1, d2 = diap
    im = im.astype(np.float32)
    im = (im - d1) / (d2 - d1)
    im[im < 0] = 0
    im[im > 1] = 1
    im = (255. * im).astype(np.uint8)
    return im


if __name__ == '__main__':
    pass