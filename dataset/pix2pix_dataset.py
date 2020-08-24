import os
import cv2
import pickle

import imageio
import numpy as np
from PIL import Image
from collections import defaultdict

import util.util as util
from util import labels

from dataset.base_dataset import BaseDataset, get_params, get_transform, normalize

import torch
import torchvision.utils as vutils
import torchvision.transforms.functional as F
import torchvision.transforms as transforms

norm = normalize()

class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt, phase='train'):
        self.opt = opt
        self.phase = phase
        
        label_paths, image_paths, instance_paths, bboxes_path = self.get_paths(opt, phase=phase)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]
        instance_paths = instance_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths
        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt, phase):
        label_paths = []
        image_paths = []
        instance_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths, instance_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        WW, HH = label.size

        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)

        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, method=Image.BICUBIC, normalize=True)
        image_tensor = transform_image(image)

        instance_tensor = 0

        input_dict = {
                    'image': image_tensor,
                    'label': label_tensor,
                    'instance': instance_tensor,
                    'path': image_path,
                    'label_path': label_path,
                    }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
