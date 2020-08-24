import os.path
import random
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as transforms

from tqdm import tqdm

from dataset.base_dataset import BaseDataset, get_params, get_transform, normalize
from dataset.pix2pix_dataset import Pix2pixDataset
from dataset.image_folder import make_dataset

from util import util

class HumanParsingDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser = Pix2pixDataset.modify_commandline_options(parser)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=18) # 17 classs + null
        parser.set_defaults(aspect_ratio=1.0)
        opt, _ = parser.parse_known_args()
        if hasattr(opt, 'num_upsampling_layers'):
            parser.set_defaults(num_upsampling_layers='more')
        return parser

    def initialize(self, opt, phase='train', label_len=-1):
        self.opt = opt
        self.phase = phase
        
        if label_len != -1:
            label_paths, image_paths, instance_paths = self.get_paths_by_label_length(opt, phase=phase, label_len=label_len)
        else:
            label_paths, image_paths, instance_paths = self.get_paths(opt, phase=phase)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths
        self.instance_paths = instance_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)

        flip_p = 0.0

        transform_label = self.get_transform(self.opt, method=Image.NEAREST, normalize=False, flip_prob=flip_p)
        label_tensor = transform_label(label) * 255.0

        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)

        image = Image.open(image_path)
        image = image.convert('RGB')

        # NOTE: if label flip, image should flip as well...
        transform_image = self.get_transform(self.opt, method=Image.BICUBIC, normalize=True, flip_prob=flip_p)
        image_tensor = transform_image(image)

        input_dict = {
                      'image': image_tensor,
                      'label': label_tensor,
                      'img_path': image_path,
                      'label_path': label_path,
                      }

        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_paths(self, opt, phase):
        root = opt.dataroot
        phase = 'val' if phase == 'test' else 'train'
 
        # find all images
        image_dir = os.path.join(root, 'img_deploy')
        image_paths = make_dataset(image_dir, recursive=True)

        # find all seg_maps
        label_dir = os.path.join(root, 'seg_deploy')
        label_paths = make_dataset(label_dir, recursive=True)

        util.natural_sort(image_paths)
        util.natural_sort(label_paths)

        # split them
        split_ratio = 0.8
        n_imgs = len(image_paths)
        all_idx = [x for x in range(n_imgs)]
        train_idx = np.random.choice(n_imgs, size=int(n_imgs * split_ratio), replace=False)
        test_idx = np.array([x for x in all_idx if x not in train_idx])

        if phase == 'train':
            label_paths = [label_paths[ix] for ix in train_idx]
            image_paths = [image_paths[ix] for ix in train_idx]
        else:
            label_paths = [label_paths[ix] for ix in test_idx]
            image_paths = [image_paths[ix] for ix in test_idx]

        assert len(set(train_idx).intersection(set(test_idx))) == 0

        instance_paths = None

        return label_paths, image_paths, instance_paths

    def get_paths_by_label_length(self, opt, phase, label_len):
        root = opt.dataroot
        phase = 'val' if phase == 'test' else 'train'
 
        # find all images
        image_dir = os.path.join(root, 'img_deploy')
        all_image_paths = make_dataset(image_dir, recursive=True)

        # find all seg_maps
        label_dir = os.path.join(root, 'seg_deploy')
        all_label_paths = make_dataset(label_dir, recursive=True)

        util.natural_sort(all_image_paths)
        util.natural_sort(all_label_paths)

        image_paths = []
        label_paths = []

        transform_label = self.get_transform(self.opt, method=Image.NEAREST, normalize=False, flip_prob=0)

        for (img_f, label_f) in tqdm(zip(all_image_paths, all_label_paths), total=len(all_label_paths)):
            assert self.paths_match(label_f, img_f)

            # need to know the label_length
            label = Image.open(label_f)
            label_tensor = transform_label(label) * 255.
            l = len(label_tensor.unique()) - 1

            if l == label_len:
                image_paths.append(img_f)
                label_paths.append(label_f)

        util.natural_sort(image_paths)
        util.natural_sort(label_paths)

        # split them
        split_ratio = 0.8
        n_imgs = len(image_paths)
        all_idx = [x for x in range(n_imgs)]
        train_idx = np.random.choice(n_imgs, size=int(n_imgs * split_ratio), replace=False)
        test_idx = np.array([x for x in all_idx if x not in train_idx])

        if phase == 'train':
            label_paths = [label_paths[ix] for ix in train_idx]
            image_paths = [image_paths[ix] for ix in train_idx]
        else:
            label_paths = [label_paths[ix] for ix in test_idx]
            image_paths = [image_paths[ix] for ix in test_idx]

        assert len(set(train_idx).intersection(set(test_idx))) == 0

        instance_paths = None

        return label_paths, image_paths, instance_paths

    def get_transform(self, opt, method=Image.BICUBIC, normalize=True, flip_prob=0.5, resize=True):
        transform_list = []

        osize = [opt.size, opt.size]
        if resize:
            transform_list.append(transforms.Resize(osize, method))

        if self.phase == 'train':
            transform_list.append(transforms.RandomHorizontalFlip(p=flip_prob))
        else:
            pass

        transform_list += [transforms.ToTensor()]

        if normalize:
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                    (0.5, 0.5, 0.5)
                                                    )]

        return transforms.Compose(transform_list)

    def paths_match(self, label_path, img_path):
        name1 = os.path.basename(label_path)[:-4] # no extension name
        name2 = os.path.basename(img_path)[:-4]   # no extension name
        return name1 == name2