#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import os
import glob
import json


def load_config_inf(path_cfg: str, find_onnx=True) -> dict:
    with open(path_cfg, 'r') as f:
        cfg = json.load(f)
    wdir_ = os.path.dirname(path_cfg)
    cfg['wdir'] = wdir_
    cfg['trn_abs'] = os.path.join(wdir_, cfg['trn'])
    cfg['val_abs'] = os.path.join(wdir_, cfg['val'])
    #
    cfg_prefig = os.path.basename(os.path.splitext(path_cfg)[0])
    cfgm = cfg['model']
    model_prefix = '{}_{}_i{}_o{}_hCNN_b{}_{}'.format(
        cfg_prefig,
        cfgm['type'], cfgm['inp'], cfgm['out'],
        int(cfgm['params']['bn']), cfgm['params']['nlin']
    )
    cfg['model_prefix'] = model_prefix
    model_dir = '{}_c{}_{}'.format(model_prefix, cfg['crop_size'], cfg['loss'])
    dir_model = os.path.join(wdir_, 'models', model_dir)
    os.makedirs(os.path.dirname(dir_model), exist_ok=True)
    cfg['dir_model'] = dir_model
    if find_onnx:
        paths = sorted(glob.glob(os.path.join(dir_model, '*.onnx')))
    else:
        paths = sorted(glob.glob(os.path.join(dir_model, '*.ckpt')))
    if len(paths) < 1:
        raise FileNotFoundError('Cant find any model in directory [{}]'.format(dir_model))
    cfg['path_model'] = paths[0]
    return cfg


def get_model_config_path(split=0) -> str:
    path_cfg = os.path.join(os.path.dirname(__file__), 'model_bka', f'cfg-p500r2-s{split}-bcedice.json')
    return path_cfg


def get_model_cfg(find_onnx=True, split=0) -> dict:
    path_cfg = get_model_config_path(split=split)
    return load_config_inf(path_cfg, find_onnx=find_onnx)


if __name__ == '__main__':
    pass