#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import cv2
import copy
from typing import Optional as O, Union as U
import numpy as np
import pandas as pd
import geopandas as gp
import matplotlib.pyplot as plt
from osgeo import gdal, ogr, osr
from torch.utils.data import Dataset, DataLoader


def _path_to_ds(ds: U[str, gdal.Dataset]) -> gdal.Dataset:
    if isinstance(ds, str):
        ds = gdal.OpenEx(ds)
    return ds


class DataCropper(Dataset):

    def __init__(self, ds_dict: dict, res_target: int = 2,
                 crop_size: int = 1024, pad_size: int = 32,
                 ds_ref=None, dtype=np.uint8, resample_alg='bilinear',
                 keys_merge=None):
        self.ds_dict = {x: _path_to_ds(y) for x, y in ds_dict.items()}
        self.res = res_target
        self.crop_size = crop_size
        self.pad_size = pad_size
        self.dtype = dtype
        self.resample_alg = resample_alg
        self.keys_merge = keys_merge
        if ds_ref is None:
            self.ds_ref = next(iter(self.ds_dict))
        else:
            self.ds_ref = ds_ref

    def build(self):
        self.ds_info = get_ds_info(self.ds_ref)
        self.grid_bboxes = get_grid_bboxes(self.ds_info, crop_size_pix=self.crop_size,
                                           pad_size_pix=self.pad_size, res_target=self.res)
        self.irc = self.grid_bboxes['img_rc']
        return self

    def __len__(self):
        return len(self.grid_bboxes['bboxes'])

    def __getitem__(self, idx) -> dict:
        bbox_info = self.grid_bboxes['bboxes'][idx]
        (x1, y1), (x2, y2) = bbox_info['bbox_m']
        imgs_crop = {}
        for ds_name, ds in self.ds_dict.items():
            ds_crop = gdal.Warp('', ds, format='MEM', outputBounds=(x1, y1, x2, y2),
                                width=self.crop_size, height=self.crop_size,
                                resampleAlg=self.resample_alg)
            img = ds_crop.ReadAsArray()
            if self.dtype is not None:
                img = img.astype(self.dtype)
            imgs_crop[ds_name] = img
        if self.keys_merge is not None:
            tmp_ = []
            for x in self.keys_merge:
                y = imgs_crop[x]
                if y.ndim < 3:
                    y = y[None, ...]
                tmp_.append(y)
            imgs_crop = np.concatenate(tmp_, axis=0)
        ret = {
            'bbox': bbox_info,
            'imgs': imgs_crop
        }
        return ret


def get_ds_info(ds: U[str, gdal.Dataset]) -> dict:
    if isinstance(ds, str):
        ds = gdal.OpenEx(ds, 0)
    sx, sy = ds.RasterXSize, ds.RasterYSize
    prj = osr.SpatialReference()
    prj.ImportFromWkt(ds.GetProjection())
    trf = ds.GetGeoTransform()
    res = float( (np.abs(trf[1]) + np.abs(trf[-1])) / 2 )
    x1, y1 = trf[0], trf[3]
    x2, y2 = x1 + sx * trf[1], y1 + sy * trf[-1]
    y1_min = min(y1, y2)
    y2_max = max(y1, y2)
    bbox = np.array([[x1, y1_min], [x2, y2_max]])
    ret = {
        'sx': sx, 'sy': sy,
        'res': res,
        'trf': trf,
        'prj': prj,
        'prj_str': prj.ExportToProj4(),
        'bbox': bbox,
        'xy': bbox[1] - bbox[0]
    }
    return ret


def get_grid_bboxes(ds_info: dict, crop_size_pix: int,
                    pad_size_pix: int, res_target=None) -> dict:
    res = ds_info['res'] if res_target is None else res_target
    crop_size_in_pix = crop_size_pix - 2 * pad_size_pix
    #
    crop_size_in_m = crop_size_in_pix * res
    pad_size_m = pad_size_pix * res
    crop_size_m = crop_size_in_m + 2 * pad_size_m
    # Y-axis direction: top -> bottom
    x0, y0 = min(ds_info['bbox'][:, 0]), max(ds_info['bbox'][:, 1])
    dxm, dym = ds_info['xy']
    nx, ny = int(np.ceil(dxm / crop_size_in_m)), int(np.ceil(dym / crop_size_in_m))
    #
    buff_size_rc = (ny * crop_size_in_pix, nx * crop_size_in_pix)
    buff_size_rc_crop = int(np.round(dym / res)), int(np.round(dxm / res))
    trf_out = list(copy.deepcopy(ds_info['trf']))
    trf_out[1] *= (dxm / buff_size_rc_crop[1]) / abs(trf_out[1])
    trf_out[-1] *= (dym / buff_size_rc_crop[0]) / abs(trf_out[-1])
    trf_out = tuple(trf_out)
    bboxes = []
    for x in range(nx):
        for y in range(ny):
            xp, yp = x * crop_size_in_pix, y * crop_size_in_pix
            xm, ym = x0 + x * crop_size_in_m, y0 - y * crop_size_in_m
            x1m, y1m = float(xm - pad_size_m), float(ym + pad_size_m)
            x2m, y2m = float(x1m + crop_size_m), float(y1m - crop_size_m)
            y1m_min = min(y1m, y2m)
            y2m_max = max(y1m, y2m)
            bbox_crop_m = [[x1m, y1m_min], [x2m, y2m_max]]
            bbox_crop_rc = [[yp, xp], [yp + crop_size_in_pix, xp + crop_size_in_pix]]
            crop_info_ = {
                'bbox_m': bbox_crop_m,
                'bbox_rc': bbox_crop_rc
            }
            bboxes.append(crop_info_)
    ret = {
        'bboxes': bboxes,
        'trf': trf_out,
        'size_rc': buff_size_rc,
        'size_rc_crop': buff_size_rc_crop,
        'img_rc': [[pad_size_pix, pad_size_pix], [pad_size_pix + crop_size_in_pix, pad_size_pix + crop_size_in_pix]]
    }
    return ret





if __name__ == '__main__':
    pass