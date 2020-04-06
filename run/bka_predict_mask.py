#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import time
import shutil
import numpy as np
import pandas as pd
from osgeo import gdal
import argparse
import logging
import onnxruntime as ort
from torch.utils.data import DataLoader

from bka_segm_api.utils import \
    _check_files, DataCropper, polygonize_pmap, \
    add_ref_to_img, generate_tmp_path
from bka_segm_api.resources \
    import get_model_cfg


def build_pmap(ort_session: ort.InferenceSession, dataset: DataCropper, jobs: int, num_print=5) -> gdal.Dataset:
    dataloader = DataLoader(dataset, batch_size=1, num_workers=jobs, collate_fn=lambda x: x)
    irc = dataset.irc
    pmap = np.zeros(dataset.grid_bboxes['size_rc'], dtype=np.uint8)
    num_ = len(dataloader)
    step_ = int(np.ceil(num_ / num_print))
    for xi, x in enumerate(dataloader):
        img_ = (x[0]['imgs'][None, ...] / 255.).astype(np.float32)
        rc = x[0]['bbox']['bbox_rc']
        pmap_crop = ort_session.run(None, {'input': img_})
        pmap_crop = (255. * pmap_crop[0][0]).astype(np.uint8)
        pmap[..., rc[0][0]:rc[1][0], rc[0][1]:rc[1][1]] = pmap_crop[..., irc[0][0]:irc[1][0], irc[0][1]:irc[1][1]]
        if (xi % step_) == 0:
            logging.info('\t\t\t(pmap) [{}/{}] <- [{}]'.format(xi, num_, pmap_crop.shape))
    siz_rc = dataset.grid_bboxes['size_rc_crop']
    pmap = pmap[..., :siz_rc[0], :siz_rc[1]]
    ds_pmap = add_ref_to_img(pmap, dataset.ds_ref)
    return ds_pmap


def main(path_pan: str, path_mul: str, path_msk: str, out: str = None,
         crop_size: int = 2048, pad: int = 32, resolution: int = None,
         export_pmap: bool = True, export_geom: bool = True, pmap_t=0.8, jobs=2):
    if not(export_pmap or export_geom):
        export_pmap = True
        export_geom = True
    # (1) prepare output paths...
    if out is not None:
        out = os.path.splitext(path_pan)[0]
    path_pmap = out + '_pmap.tif'
    path_geom = out + '_bndr.gpkg'
    # (2) check output exists:
    is_ok = []
    is_ok += [] if not export_pmap else [os.path.isfile(path_pmap)]
    is_ok += [] if not export_geom else [os.path.isfile(path_geom)]
    if np.sum(is_ok) == len(is_ok):
        logging.warning('\t!!! ouput files exists, skip...\n\t- [{}]\n\t- [{}]'
                        .format(path_pmap, path_geom))
        return
    #
    cfg = get_model_cfg(find_onnx=True)
    path_model_onnx = cfg['path_model']
    _check_files([path_pan, path_mul, path_model_onnx])
    ds_dict = {
        'pan': path_pan,
        'mul': path_mul
    }
    logging.info('(1) :: data-crop')
    data_crop = DataCropper(ds_dict=ds_dict, res_target=resolution,
                            crop_size=crop_size, pad_size=pad,
                            ds_ref=ds_dict['pan'], keys_merge=('pan', 'mul')).build()
    logging.info('\t\t... done, #grid-size = {}'.format(len(data_crop.grid_bboxes['bboxes'])))
    logging.info('(2) :: create onnx inrerence runtime session')
    ort_session = ort.InferenceSession(path_model_onnx)
    logging.info('\t\t... done, session = {}'.format(ort_session))
    t = time.time()
    logging.info('(3) :: [build-pmap] out size ~ {}'.format(data_crop.grid_bboxes['size_rc']))
    ds_pmap = build_pmap(ort_session, data_crop, jobs=jobs)
    dt = time.time() - t
    logging.info('\t\t... done, dt ~ {:0.2f} (s)'.format(dt))
    #
    if export_pmap and not os.path.isfile(path_pmap):
        logging.info(f'\t::(pmap) :: export to [{path_pmap}]')
        path_pmap_tmp = generate_tmp_path(path_pmap)
        gdal.Translate(path_pmap_tmp, ds_pmap, creationOptions=['COMPRESS=LZW'])
        shutil.move(path_pmap_tmp, path_pmap)
    if export_geom and not os.path.isfile(path_geom):
        path_geom_tmp = generate_tmp_path(path_geom)
        logging.info(f'\t::(pmap) :: export to [{path_pmap}]')
        pmap_tu8 = int(pmap_t * 255)
        ds_geom = polygonize_pmap(ds_pmap, pmap_threshold=131, path_out=path_geom_tmp)
        shutil.move(path_geom_tmp, path_geom)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--pan', required=True, default=None, type=str,
                        help='path to PAN file (BKA)')
    parser.add_argument('-m', '--mul', required=True, default=None, type=str,
                        help='path to PAN file (BKA)')
    parser.add_argument('--mask', required=False, default=None, type=str,
                        help='path to Mask file')
    parser.add_argument('-r', '--res', required=False, default=2, type=int,
                        help='output resolution, if None -> original PAN/MUL resolutions')
    parser.add_argument('--crop', required=False, default=2048, type=int,
                        help='crop size for grid-inference on big images')
    parser.add_argument('--pad', required=False, default=32, type=int,
                        help='pad size for grid-inference on big images')
    parser.add_argument('-o', '--out', required=False, default=None, type=str,
                        help='output data prefix')
    parser.add_argument('--pmap', action='store_true', help='generate PMAP as output')
    parser.add_argument('--geom', action='store_true', help='generate vector mask as output')
    parser.add_argument('-t', '--pmap_threshold', required=False, default=0.8, type=float,
                        help='threshold for probability map')
    parser.add_argument('-j', '--jobs', required=False, default=0, type=int, help='#workers')
    args = parser.parse_args()
    #
    logging.info(f'args = {args}')
    main(
        path_pan=args.pan,
        path_mul=args.mul,
        path_msk=args.mask,
        resolution=args.res,
        crop_size=args.crop,
        pad=args.pad,
        out=args.out,
        pmap_t=args.pmap_threshold,
        export_pmap=args.pmap,
        export_geom=args.geom,
        jobs=args.jobs
    )
