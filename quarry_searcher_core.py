#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
from osgeo import gdal, osr



band_names = ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B11", "B12", "B8A", "B01", "B09"]

bands_diap={
    "B01": (1091.0, 1686.0),
    "B02": (790.0, 1523.0),
    "B03": (588.0, 1458.0),
    "B04": (346.0, 1575.0),
    "B05": (322.0, 1734.0),
    "B06": (250.0, 3788.0),
    "B07": (229.0, 5070.0),
    "B08": (187.0, 5009.0),
    "B8A": (172.0, 5480.0),
    "B09": (66.0,  1873.0),
    "B10": (5.0,   44.0),
    "B11": (41.0,  3035.0),
    "B12": (23.0,  2511.)
}


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


def generate_tmp_path(path, root=None, tmp='.tmp'):
    if root is None:
        root = os.path.dirname(path)
    rel_ = os.path.relpath(path, root)
    ret = os.path.join(root, tmp, rel_)
    os.makedirs(os.path.dirname(ret), exist_ok=True)
    return ret


def add_ref_to_img(imgu8: np.ndarray, ds_ref) -> gdal.Dataset:
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
    srs.ImportFromWkt(ds_ref.GetProjection())
    ds_out.SetProjection(srs.ExportToWkt())
    ds_out.SetGeoTransform(ds_ref.GetGeoTransform())
    if imgu8.ndim < 3:
        imgu8 = [imgu8]
    for bi, b in enumerate(imgu8):
        ds_band = ds_out.GetRasterBand(bi + 1)
        ds_band.WriteArray(b)
        ds_band.FlushCache()
    ds_out.FlushCache()
    return ds_out


class ModelInference():

    def __init__(self, path_model,
                 inp_name: str = 'input_1',
                 out_name: str = 'lambda_1/Reshape', grow_gpu_memory=False):
        self.path_model = path_model
        self.inp_name = inp_name
        self.out_name = out_name
        self.grow_gpu_memory = grow_gpu_memory
        self._graph = None
        self._sess = None

    def build(self):
        tf.reset_default_graph()
        self._graph = tf.Graph()
        session_config = None
        if self.grow_gpu_memory:
            session_config = tf.ConfigProto()
            session_config.gpu_options.allow_growth = True
        self._sess = tf.Session(graph=self._graph, config=session_config)
        with self._graph.as_default():
            with tf.gfile.GFile(self.path_model, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name='')
        #
        if (self.inp_name is None) or (self.out_name is None):
            op_names = [x.name for x in self._graph.get_operations()]
            self.inp_name = op_names[0]
            self.out_name = op_names[-1]
        logging.info('model inp/out names: ({})/({})'.format(self.inp_name, self.out_name))
        inp_name_op = self.inp_name + ':0'
        out_name_op = self.out_name + ':0'
        self._inp_op = self._graph.get_tensor_by_name(inp_name_op)
        self._out_op = self._graph.get_tensor_by_name(out_name_op)
        return self

    def inference(self, img_batch: np.ndarray) -> np.ndarray:
        feed_dict = {self._inp_op: img_batch}
        ret = self._sess.run(self._out_op, feed_dict=feed_dict)
        return ret


#############################################
def pad_image_for_batches(img, crop_size=2048, pad_size=64, mode='reflect') -> (np.ndarray, tuple):
    nrc0 = np.array(img.shape[:2])
    num_rc = np.ceil(nrc0 / crop_size).astype(np.int)
    nrc1 = (crop_size * num_rc).astype(np.int)
    nrc_diff = (nrc1 - nrc0) + pad_size
    img_pad = np.pad(img, [[pad_size, nrc_diff[0]], [pad_size, nrc_diff[1]], [0, 0]], mode=mode)
    return img_pad, tuple(num_rc)


def pad_image_indices_generator(img_shape, crop_size=192, pad_size=64):
    img_shape = np.array(img_shape[:2])
    num_rc = np.ceil(img_shape / crop_size).astype(np.int32)
    for r in range(num_rc[0]):
        for c in range(num_rc[1]):
            r0 = r * crop_size + pad_size
            c0 = c * crop_size + pad_size
            bbox = ((r0 - pad_size, c0 - pad_size), (r0 + crop_size + pad_size, c0 + crop_size + pad_size))
            bbox_crop = ((r0, c0), (r0 + crop_size, c0 + crop_size))
            yield bbox, bbox_crop


def pad_image_indices(img_shape, crop_size=192, pad_size=64):
    return list(pad_image_indices_generator(img_shape, crop_size, pad_size))


def infer_pmap_by_grid(model: ModelInference, img: np.ndarray, crop_size=256, pad_size=64,
                       is_norm_u8=True, num_print=None, pproc_fun=None):
    img_dsc_shape = img.shape[:2]
    crop_size_pad = crop_size - 2 * pad_size
    img_dsc_pad, _ = pad_image_for_batches(img, crop_size=crop_size_pad, pad_size=pad_size)
    rc_bboxes = pad_image_indices(img_dsc_shape, crop_size=crop_size_pad, pad_size=pad_size)
    pmap = None
    num_bboxes = len(rc_bboxes)
    if num_print is not None:
        step_ = int(np.ceil(num_bboxes / num_print))
    for ii, (bbox, bbox_crop) in enumerate(rc_bboxes):
        img_batch = img_dsc_pad[None, bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], ...]
        # predict = img_batch[0][..., 0]
        if pproc_fun is not None:
            img_batch = pproc_fun(img_batch)
        predict = model.inference(img_batch)[0]
        if pmap is None:
            if predict.ndim < 3:
                pmap = np.zeros(list(img_dsc_pad.shape[:2]), dtype=np.float32)
            else:
                pmap = np.zeros(list(img_dsc_pad.shape[:2]) + [predict.shape[-1]], dtype=np.float32)
        if pad_size > 0:
            pmap[bbox_crop[0][0]:bbox_crop[1][0], bbox_crop[0][1]:bbox_crop[1][1], ...] = predict[pad_size:-pad_size, pad_size:-pad_size, ...]
        else:
            pmap[bbox_crop[0][0]:bbox_crop[1][0], bbox_crop[0][1]:bbox_crop[1][1], ...] = predict[:, :, ...]
        if (num_print is not None) and ((ii % step_) == 0):
            logging.info('\t\t(infer-pmap) [{}/{}]'.format(ii, num_bboxes))
    pmap = pmap[pad_size:img_dsc_shape[0] + pad_size, pad_size:img_dsc_shape[1] + pad_size, ...]
    if is_norm_u8:
        pmap = (255 * pmap).astype(np.uint8)
    return pmap


def main_test_grid_inference():
    img = np.random.uniform(0, 1, (4100, 3700, 3)).astype(np.float32)
    pad_size = 64
    crop_size = 1024
    model = None
    pmap = infer_pmap_by_grid(
        model=model,
        img=img,
        crop_size=crop_size,
        pad_size=pad_size,
        is_norm_u8=False
    )
    print('-')


if __name__ == '__main__':
    main_test_grid_inference()