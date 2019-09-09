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
from quarry_searcher_core import generate_tmp_path


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
    ds_out.SetProjection(crs_xy)
    #
    ds_geom = gdal.OpenEx(geom_bl.to_json())
    gdal.RasterizeLayer(ds_out, [1], ds_geom.GetLayer(), burn_values=[1], options=['ALL_TOUCHED=TRUE'])
    return ds_out


def main_clip_raster(path_img: str, path_geom: str, pad: int, res: int,
                     path_out: str = None, attr: str = None,
                     no_split=False, add_mask=False):
    bn_img = os.path.splitext(os.path.basename(path_img))[0]
    bn_geom = os.path.splitext(os.path.basename(path_geom))[0]
    #
    idx_pref = bn_geom
    if path_out is None:
        path_out = os.path.splitext(path_img)[0]
        if no_split:
            idx_pref = 'clip'
    else:
        idx_pref = None
    #
    geom = gp.read_file(path_geom)
    geom = __reproject_to_img(geom, path_img)
    geom_bboxes = get_geom_bboxes(geom, pad, idx_def=idx_pref, attr=attr, no_split=no_split)
    #
    num = len(geom_bboxes)
    for xi, x in enumerate(geom_bboxes):
        g = x['geom']
        bbox_xy = x['bbox']
        fidx = x['idx']
        if fidx is not None:
            pout_img = path_out + '_' + fidx + '.tif'
            pout_msk = path_out + '_' + fidx + '_msk.tif'
        else:
            pout_img = path_out + '.tif'
            pout_msk = path_out + '_msk.tif'
        if os.path.isfile(pout_img):
            logging.warning(f'\t!!! output file exist, skip ... [{pout_img}]')
        if add_mask:
            logging.info('\t\t({}/{}) (1) (rasterize-mask) #plgn={}'.format(xi, num, len(g)))
            ds_msk = rasterize_geom_mask(geom_xy=g, bbox_xy=bbox_xy, res=res)
        logging.info('\t\t({}/{}) (2) (crop-image) size-m = {:0.2f}x{:0.2f}, res={:0.2f} (m/pix)'
                     .format(xi, num, bbox_xy[1][0] - bbox_xy[0][0], bbox_xy[1][1] - bbox_xy[0][1], res))
        ds_img = gdal.Warp('', path_img, format='MEM',
                           outputBounds=np.array(bbox_xy).reshape(-1).tolist(),
                           xRes=res, yRes=res,
                           resampleAlg='bilinear')
        pout_img_tmp = generate_tmp_path(pout_img)
        pout_msk_tmp = generate_tmp_path(pout_msk)
        if add_mask:
            logging.info('\t({}/{}) mask -> ({})'.format(xi, num, pout_msk))
            gdal.Translate(pout_msk_tmp, ds_msk, creationOptions=['COMPRESS=LZW', 'NBITS=1'])
            shutil.move(pout_msk_tmp, pout_msk)
        logging.info('\t({}/{}) crop-img -> ({})'.format(xi, num, pout_img))
        gdal.Translate(pout_img_tmp, ds_img, creationOptions=['COMPRESS=LZW'])
        shutil.move(pout_img_tmp, pout_img)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img',  required=True, type=str, default=None, help='path to geo-image')
    parser.add_argument('-g', '--geom', required=True, type=str, default=None, help='path to geometry')
    parser.add_argument('-a', '--attr', required=False, type=str, default=None, help='attribute for geometry combining')
    parser.add_argument('-r', '--resolution', required=False, type=str, default=5, help='target resolution')
    parser.add_argument('-p', '--pad', required=False, type=str, default=200,  help='extra padding for geometry clip')
    parser.add_argument('-o', '--out', required=False, type=str, default=None,
                        help='output filename-prefix (for no-split)')
    parser.add_argument('--no_split', action='store_true', help='generate separate pmap for every polygon from geom')
    parser.add_argument('--no_mask',  action='store_true', help='do not generate geometry mask')
    args = parser.parse_args()
    logging.info('args = {}'.format(args))
    #
    main_clip_raster(
        path_img=args.img,
        path_geom=args.geom,
        attr=args.attr,
        pad=args.pad,
        res=args.resolution,
        path_out=args.out,
        no_split=args.no_split,
        add_mask=not args.no_mask
    )