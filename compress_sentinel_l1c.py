#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import glob
import time
import shutil
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import logging
import argparse
from quarry_searcher_core import generate_tmp_path, band_names,\
    bands_diap, map_u16_to_u8, add_ref_to_img
from osgeo import gdal, osr


def __get_mapped_inp_bands(dir_inp: str) -> dict:
    ret = OrderedDict()
    bad_files = []
    for bi, bn in enumerate(band_names):
        path_find = os.path.join(dir_inp, f'*_{bn}.jp2')
        try:
            path_ = glob.glob(path_find)[0]
            ret[bn] = path_
        except FileNotFoundError as err:
            logging.error('\t\tcant find band [{}] by reg-exp [{}], err=[{}]'.format(bn, path_find, err))
            bad_files.append(bn)
    if len(bad_files) > 0:
        raise FileNotFoundError('Cant find bands {}'.format(bad_files))
    return ret


def __load_and_map_s2_raw(path_map: dict, bands_diap: dict, res: int = 10) -> np.ndarray:
    ret = None
    num = len(path_map)
    t1 = time.time()
    logging.info('\t.. loading/converting raw s2-data into memory, #img={}'.format(num))
    for bi, (bn, bp) in enumerate(path_map.items()):
        logging.info('\t\t[{}/{}] <- [{}]'.format(bi, num, bp))
        ds_ = gdal.Translate('', bp, format='MEM', xRes=res, yRes=res, resampleAlg='average')
        im_ = ds_.ReadAsArray()
        del ds_
        im_ = map_u16_to_u8(im_, bands_diap[bn])
        if ret is None:
            ret = np.zeros((num, ) + im_.shape[:2], dtype=np.uint8)
        ret[bi] = im_
    dt = time.time() - t1
    logging.info('\t... done, dt ~ {:0.2f} (s)'.format(dt))
    return ret


def main_compress_s2_l1c(dir_inp: str, path_out: str = None, res: int = 10):
    if path_out is None:
        path_out = os.path.join(dir_inp, 's2_u8.jp2')
    if os.path.isfile(path_out):
        logging.warning('\t!!! output file exist, skip... [{}]'.format(path_out))
        return
    map_path = __get_mapped_inp_bands(dir_inp)
    imgs = __load_and_map_s2_raw(map_path, bands_diap, res=res)
    ds_out =  add_ref_to_img(imgs, map_path['B08'])
    logging.info('\tadd ref and export to: [{}]'.format(path_out))
    path_out_tmp = generate_tmp_path(path_out)
    gdal.Translate(path_out_tmp, ds_out, format='JP2OpenJPEG')
    shutil.move(path_out_tmp, path_out)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dir_inp', required=True,  type=str, default=None, help='input directory with RAW S2-L1C data')
    parser.add_argument('-o', '--out', required=False, type=str, default=None, help='output directory with compressed S2-L1C data')
    parser.add_argument('-r', '--resolution', required=False, type=int, default=10, help='output resolution for data')
    args = parser.parse_args()
    #
    main_compress_s2_l1c(
        dir_inp=args.dir_inp,
        path_out=args.out,
        res=args.resolution
    )

