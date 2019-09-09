#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import time
import shutil
import numpy as np
import pandas as pd
import geopandas as gp
from quarry_searcher_core import generate_tmp_path, ModelInference,\
    read_img_into_memory, infer_pmap_by_grid, add_ref_to_img
from osgeo import gdal, ogr, osr
import logging
import argparse
import json

import matplotlib.pyplot as plt



def load_model_config(path_cfg: str) -> dict:
    with open(path_cfg, 'r') as f:
        cfg = json.load(f)
    wdir =  os.path.dirname(path_cfg)
    cfg['dir'] = wdir
    cfg['path_model_abs'] = os.path.join(wdir, cfg['model'])
    cfg['inp_op_name'] = cfg['inp_name'] + ':0'
    cfg['out_op_name'] = cfg['out_name'] + ':0'
    return cfg


def __get_img_pix_size_xy(path_img: str) -> tuple:
    ds_ = gdal.OpenEx(path_img)
    sx = ds_.RasterXSize
    sy = ds_.RasterYSize
    ret = (sx, sy)
    return ret


def build_pmap_single(model: ModelInference, path_img: str, pad_size: int, to_u8=True) -> np.ndarray:
    size_img = __get_img_pix_size_xy(path_img)[::-1]
    img_pad = read_img_into_memory(path_img, ret_batch=True, pad_size=pad_size, remove_last_channel=False)
    pmap = model.inference(img_pad)[0]
    pmap = pmap[:size_img[0], :size_img[1], ...]
    if to_u8:
        pmap = (255. * pmap).astype(np.uint8)
    return pmap


def pproc_fun_for_pmap(img: np.ndarray, coef=255.) -> np.ndarray:
    ret = img.astype(np.float32) / coef
    return ret


def build_pmap_grid(model: ModelInference, path_img: str, pad_size: int, crop_size: int, to_u8=True) -> np.ndarray:
    img = read_img_into_memory(path_img,
                               ret_batch=False,
                               pad_size=None,
                               norm_coef=None,
                               remove_last_channel=False)
    pmap = infer_pmap_by_grid(model, img,
                              crop_size=crop_size,
                              pad_size=pad_size,
                              is_norm_u8=to_u8,
                              num_print=10,
                              pproc_fun=lambda x: x.astype(np.float32) / 255.)
    return pmap


def __polygonize_pmap(pmap_ds: gdal.Dataset, pmap_threshold=131, layer_name='unknown.gpkg', path_out=None) -> gdal.Dataset:
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


def main_inference_pmap(path_img: str, path_model_cfg: str, path_out: str) -> bool:
    cfg = load_model_config(path_model_cfg)
    model = ModelInference(path_model=cfg['path_model_abs'],
                           inp_name=cfg['inp_name'],
                           out_name=cfg['out_name'],
                           grow_gpu_memory=True).build()
    if path_out is None:
        path_out = os.path.splitext(path_img)[0] + '_pmap.tif'
    path_out_vec = os.path.splitext(path_out)[0] + '_vec.gpkg'
    if os.path.isfile(path_out) and os.path.isfile(path_out_vec):
        logging.warning('\t!!! output pmap exist, skip... ({})'.format(path_out))
        return True
    min_size = min(__get_img_pix_size_xy(path_img))
    t1 = time.time()
    logging.info(f'\tstart probability map inference for ({path_img})')
    if min_size < cfg['crop_size']:
        pmap = build_pmap_single(model, path_img, pad_size=cfg['pad_size'])
    else:
        pmap = build_pmap_grid(model, path_img, pad_size=cfg['pad_size'], crop_size=cfg['crop_size'], to_u8=True)
    dt = time.time() - t1
    logging.info(f'\t... done, dt ~ {dt:0.2f} (s), pmap-shape={pmap.shape}')
    #
    path_msk = os.path.splitext(path_img)[0] +'_msk.tif'
    if os.path.isfile(path_msk):
        msk = gdal.OpenEx(path_msk).ReadAsArray()
        pmap[msk < 0.5] = 0
    path_out_tmp = generate_tmp_path(path_out)
    path_out_vec_tmp = generate_tmp_path(path_out_vec)
    #
    ds_pmap = add_ref_to_img(pmap, path_img)
    ds_pmap_vec = __polygonize_pmap(ds_pmap, path_out=path_out_vec_tmp)
    #
    logging.info(f'\texport pmap/pmap-vec data -> ({path_out_vec})')
    # gdal.VectorTranslate(path_out_vec_tmp, ds_pmap_vec, format='GPKG')
    gdal.Translate(path_out_tmp, ds_pmap, creationOptions=['COMPRESS=LZW'])
    shutil.move(path_out_vec_tmp, path_out_vec)
    shutil.move(path_out_tmp, path_out)
    return True


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', required=True, type=str, default=None, help='path to s2 u8-compressed file')
    parser.add_argument('-c', '--model_cfg', required=True, type=str, default=None, help='path to model config in json-format')
    parser.add_argument('-o', '--out', required=False, type=str, default=None, help='outout probability map path')
    parser.add_argument('-t', '--threshold', required=False, type=str, default=None, help='threshold for pmap polygonization')
    parser.add_argument('--no_split', action='store_true', help='generate separate pmap for every polygon from geom')
    args = parser.parse_args()
    #
    logging.info('args = {}'.format(args))
    main_inference_pmap(
        path_img=args.img,
        path_model_cfg=args.model_cfg,
        path_out=args.out
    )