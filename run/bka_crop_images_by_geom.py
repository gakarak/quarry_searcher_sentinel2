#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import fire
import shutil
import numpy as np
import logging
from typing import Union as U, Optional as O
import geopandas as gp
from osgeo import gdal, ogr, osr
#
from bka_segm_api.utils import generate_tmp_path, \
    __reproject_to_img, get_geom_bboxes, \
    rasterize_geom_mask, get_ds_resolution, \
    _check_files


def main(path_pan: str, path_mul: str, path_geom: str,
         no_split: bool = False, pad_m: int = 1000,
         resolution: int = None, add_mask: bool = True,
         attr: str = None, out: str = None):
    _check_files([path_pan, path_mul, path_geom])
    #
    # bn_img_pan = os.path.splitext(os.path.basename(path_pan))[0]
    # bn_img_mul = os.path.splitext(os.path.basename(path_mul))[0]
    bn_geom = os.path.splitext(os.path.basename(path_geom))[0]
    idx_pref = bn_geom
    if out is None:
        path_out_pan = os.path.splitext(path_pan)[0]
        path_out_mul = os.path.splitext(path_mul)[0]
        if no_split:
            idx_pref = 'clip'
    else:
        idx_pref = None
        path_out_pan = out + '_PAN'
        path_out_mul = out + '_MUL'
    geom = gp.read_file(path_geom)
    geom = __reproject_to_img(geom, path_pan)
    geom_bboxes = get_geom_bboxes(geom, pad_m, idx_def=idx_pref, attr=attr, no_split=no_split)
    #
    if resolution is None:
        res = get_ds_resolution(path_pan)
    else:
        res = resolution
    num = len(geom_bboxes)
    for xi, x in enumerate(geom_bboxes):
        g = x['geom']
        bbox_xy = x['bbox']
        fidx = x['idx']
        if fidx is not None:
            pout_img_pan = path_out_pan + '_' + fidx + '.tif'
            pout_img_mul = path_out_mul + '_' + fidx + '.tif'
            pout_msk = path_out_pan + '_' + fidx + '_msk.tif'
        else:
            pout_img_pan = path_out_pan + '.tif'
            pout_img_mul = path_out_mul + '.tif'
            pout_msk = path_out_pan + '_msk.tif'
        if os.path.isfile(pout_img_pan):
            logging.warning(f'\t!!! output file exist, skip ... [{pout_img_pan}]')
            continue
        if add_mask:
            logging.info('\t\t({}/{}) (1) (rasterize-mask) #plgn={}'.format(xi, num, len(g)))
            ds_msk = rasterize_geom_mask(geom_xy=g, bbox_xy=bbox_xy, res=res)
        else:
            ds_msk = None
        logging.info('\t\t({}/{}) (2) (crop-image) size-m = {:0.2f}x{:0.2f}, res={:0.2f} (m/pix)'
                     .format(xi, num, bbox_xy[1][0] - bbox_xy[0][0], bbox_xy[1][1] - bbox_xy[0][1], res))
        bounds_ = np.array(bbox_xy).reshape(-1).tolist()
        if resolution is not None:
            ds_pan = gdal.Warp('', path_pan, format='MEM', outputBounds=bounds_, xRes=res, yRes=res, resampleAlg='bilinear')
            ds_mul = gdal.Warp('', path_mul, format='MEM', outputBounds=bounds_, xRes=res, yRes=res, resampleAlg='bilinear')
        else:
            ds_pan = gdal.Warp('', path_pan, format='MEM', outputBounds=bounds_, resampleAlg='bilinear')
            ds_mul = gdal.Warp('', path_mul, format='MEM', outputBounds=bounds_, resampleAlg='bilinear')
        pout_img_pan_tmp = generate_tmp_path(pout_img_pan)
        pout_img_mul_tmp = generate_tmp_path(pout_img_mul)
        pout_msk_tmp = generate_tmp_path(pout_msk)
        if add_mask:
            logging.info('\t({}/{}) mask -> ({})'.format(xi, num, pout_msk))
            gdal.Translate(pout_msk_tmp, ds_msk, creationOptions=['COMPRESS=LZW', 'NBITS=1'])
            shutil.move(pout_msk_tmp, pout_msk)
        logging.info('\t({}/{}) crop-img -> ({})'.format(xi, num, pout_img_pan))
        gdal.Translate(pout_img_mul_tmp, ds_mul, creationOptions=['COMPRESS=LZW'])
        shutil.move(pout_img_mul_tmp, pout_img_mul)
        gdal.Translate(pout_img_pan_tmp, ds_pan, creationOptions=['COMPRESS=LZW'])
        shutil.move(pout_img_pan_tmp, pout_img_pan)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)