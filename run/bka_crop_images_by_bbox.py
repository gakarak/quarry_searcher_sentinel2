#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import time
import shutil
import numpy as np
import logging
import argparse
from typing import Union as U, Optional as O
import geopandas as gp
from osgeo import gdal, ogr, osr
#
from bka_segm_api.utils import \
    generate_tmp_path, _check_files


def parse_bbox_str(bbox_str: str) -> np.ndarray:
    bbox_ = bbox_str.split(',')
    if len(bbox_) < 4:
        raise Exception(f'Invalid crop bbox: [{bbox_str}]')
    x1, y1, x2, y2 = [float(x) for x in bbox_[:4]]
    if x2 < x1:
        x2, x1 = x1, x2
    if y2 < y1:
        y2, y1 = y1, y2
    ret = np.array([[x1, y1], [x2, y2]])
    return ret


def main(path_pan: str, path_mul: str, bbox_bl: np.ndarray,
         resolution: int = None, out: str = None):
    _check_files([path_pan, path_mul])
    if out is None:
        path_out_pan = os.path.splitext(path_pan)[0] + '_clip_bbox.' + os.path.splitext(path_pan)[-1]
        path_out_mul = os.path.splitext(path_mul)[0] + '_clip_bbox.' + os.path.splitext(path_mul)[-1]
    else:
        path_out_pan = out + '_PAN.tif'
        path_out_mul = out + '_MUL.tif'
    #
    t1 = time.time()
    logging.info('\t:: crop-img -> ({})'.format(path_out_pan))
    bounds_bl = np.array(bbox_bl).reshape(-1).tolist()
    if resolution is not None:
        ds_pan = gdal.Warp('', path_pan, format='MEM', outputBounds=bounds_bl, outputBoundsSRS='EPSG:4326',
                           xRes=resolution, yRes=resolution, resampleAlg='bilinear')
        ds_mul = gdal.Warp('', path_mul, format='MEM', outputBounds=bounds_bl, outputBoundsSRS='EPSG:4326',
                           xRes=resolution, yRes=resolution, resampleAlg='bilinear')
    else:
        ds_pan = gdal.Warp('', path_pan, format='MEM', outputBounds=bounds_bl, outputBoundsSRS='EPSG:4326', resampleAlg='bilinear')
        ds_mul = gdal.Warp('', path_mul, format='MEM', outputBounds=bounds_bl, outputBoundsSRS='EPSG:4326', resampleAlg='bilinear')
    pout_img_pan_tmp = generate_tmp_path(path_out_pan)
    pout_img_mul_tmp = generate_tmp_path(path_out_mul)
    gdal.Translate(pout_img_mul_tmp, ds_mul, creationOptions=['COMPRESS=LZW'])
    shutil.move(pout_img_mul_tmp, path_out_mul)
    gdal.Translate(pout_img_pan_tmp, ds_pan, creationOptions=['COMPRESS=LZW'])
    shutil.move(pout_img_pan_tmp, path_out_pan)
    dt = time.time() - t1
    logging.info('\t\t... done, dt ~ {:0.2f} (s)'.format(dt))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pan', required=True, default=None, type=str,
                        help='path to PAN file (BKA)')
    parser.add_argument('-m', '--mul', required=True, default=None, type=str,
                        help='path to PAN file (BKA)')
    parser.add_argument('-b', '--bbox', required=True, default=None, type=str,
                        help='crop-bbox (WGS84) in comma-separated format lon_min,lat_min, lon_max, lat_max'
                             '\tfor example: --bbox=27.28737,53.75080,27.37857,53.85420')
    parser.add_argument('-r', '--res', required=False, default=None, type=int,
                        help='output resolution, if None -> original PAN/MUL resolutions')
    parser.add_argument('-o', '--out', required=False, default=None, type=str, help='output data prefix')
    args = parser.parse_args()
    logging.info(f'args = {args}')
    #
    main(
        path_pan=args.pan,
        path_mul=args.mul,
        bbox_bl=parse_bbox_str(args.bbox),
        resolution=args.res,
        out=args.out
    )