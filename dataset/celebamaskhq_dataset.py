import os.path
import random

from PIL import Image

from dataset.base_dataset import BaseDataset
from dataset.pix2pix_dataset import Pix2pixDataset
from dataset.image_folder import make_dataset

from util import util
import numpy as np

import torch
import torchvision.transforms as transforms

class CelebAMaskDatasetHQ(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser):
        parser = Pix2pixDataset.modify_commandline_options(parser)
        parser.set_defaults(preprocess_mode='fixed')
        parser.set_defaults(display_winsize=256)
        parser.set_defaults(label_nc=19) # 18 classs + null
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

        h, w, c = self.opt.size, self.opt.size, self.opt.label_nc
        label_tensor = torch.zeros(c, h, w)
        part_names = os.listdir(label_path)

        flip_p = 0.0

        # parts' map have to be all flipped or all not to be flipped
        transform_label = self.get_transform(self.opt, method=Image.NEAREST, normalize=False, flip_prob=flip_p)
        transform_label_noflip = self.get_transform(self.opt, method=Image.NEAREST, normalize=False, flip_prob=0.0)
        
        for i, l in enumerate(part_names):
            cls_idx = int(l.replace('.png', ''))
            part_path = os.path.join(label_path, l)
            label = Image.open(part_path)
            l_tensor = transform_label(label).squeeze(0)
            label_tensor[cls_idx] = l_tensor

        image_path = self.image_paths[index]
        assert self.paths_match(label_path, image_path), \
            "The label_path %s and image_path %s don't match." % \
            (label_path, image_path)

        image = Image.open(image_path)
        image = image.convert('RGB')

        # NOTE: if the labels flip, images should flip as well
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
        """ for celeba, we split train test on the fly. """

        # root: 'server'_data/opt.dataset
        root = opt.dataroot
        phase = 'val' if phase == 'test' else 'train'
 
        # find all images
        image_dir = os.path.join(root, 'img')
        image_paths = make_dataset(image_dir, recursive=True)

        # find all seg_maps
        label_dir = os.path.join(root, 'segmentation_map')
        label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]

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
        image_dir = os.path.join(root, 'img')
        label_dir = os.path.join(root, 'segmentation_map')

        image_paths = []
        label_paths = []
        
        all_image_paths = make_dataset(image_dir, recursive=True)
        all_label_paths = [os.path.join(label_dir, f) for f in os.listdir(label_dir)]

        util.natural_sort(all_image_paths)
        util.natural_sort(all_label_paths)

        for (img_f, label_f) in zip(all_image_paths, all_label_paths):
            assert self.paths_match(label_f, img_f)

            # get label-set and calculate length
            parts_path = os.listdir(label_f)
            if '0.png' in parts_path:
                import pdb; pdb.set_trace()
            l = len(parts_path)
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

        transform_list += [transforms.ToTensor()]

        if normalize:
            transform_list += [
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))
                              ]

        return transforms.Compose(transform_list)

    def paths_match(self, label_path, img_path):
        name1 = os.path.basename(label_path)
        name2 = '%05d' % int(os.path.basename(img_path).replace('.jpg', ''))
        return name1 == name2