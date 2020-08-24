import argparse
import dill as pickle
import os
import re
import importlib
from argparse import Namespace
from collections import namedtuple

import numpy as np
import imageio
from PIL import Image

import torch

## celebamaskhq
# class dict
celeba_idx_to_cls = {
    1: 'skin',
    2: 'hair',
    3: 'cloth',
    4: 'l_brow',
    5: 'r_brow',
    6: 'l_ear',
    7: 'r_ear',
    8: 'ear_r',
    9: 'l_eye',
    10: 'r_eye',
    11: 'eye_g',
    12: 'l_lip',
    13: 'u_lip',
    14: 'mouth',
    15: 'neck',
    16: 'neck_l',
    17: 'nose',
    18: 'hat',
}
n_celeba_classes = len(celeba_idx_to_cls) + 1 # 18 + null
celeba_order = [1, 15, 2, 4, 5, 6, 7, 9, 10, 17, 12, 13, 14, 18, 3, 11, 8, 16] # face, hair first --> organ --> acce.

## humanparsing
humanparsing_idx_to_cls = {
    1: 'hat',
    2: 'hair', #
    3: 'sunglass',
    4: 'upper-clothes', #
    5: 'skirt', #
    6: 'pants', #
    7: 'dress', #
    8: 'belt',
    9: 'left-shoe', #
    10: 'right-shoe', #
    11: 'face', #
    12: 'left-leg', #
    13: 'right-leg', #
    14: 'left-arm', #
    15: 'right-arm', #
    16: 'bag',
    17: 'scarf',
}

humanparsing_cls_to_idx = {v: k for k, v in humanparsing_idx_to_cls.items()}

n_humanparsing_classes = len(humanparsing_idx_to_cls) + 1 # 18 + null
humanparsing_order = [11, 2, 14, 15, 12, 13, 4, 7, 5, 6, 9, 10, 1, 3, 8, 17, 16] # top-down, body -> clothes -> acces.