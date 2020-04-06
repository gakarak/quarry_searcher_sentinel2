#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
from typing import Union as U, \
    Optional as O

from .core import add_ref_to_img, \
    rasterize_geom_mask, get_ds_resolution, \
    get_geom_bboxes, generate_tmp_path, \
    pad_img_to_cropsize, read_img_into_memory, \
    __reproject_to_img

from .gis import polygonize_pmap

from .improc_big import DataCropper, \
    get_ds_info, get_grid_bboxes


def _check_files(paths: U[str, list]):
    if isinstance(paths, str):
        paths = [paths]
    for x in paths:
        if not os.path.isfile(x):
            raise FileNotFoundError('Cant find file [{}]'.format(x))

if __name__ == '__main__':
    pass