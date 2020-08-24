import argparse
import dill as pickle
import os
import re
import importlib
import socket

from argparse import Namespace
from collections import namedtuple

import numpy as np
import imageio
from PIL import Image

import torch

def save_obj(obj, name):
    with open(name, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name, 'rb') as f:
        return pickle.load(f)

def copyconf(default_opt, **kwargs):
    conf = argparse.Namespace(**vars(default_opt))
    for key in kwargs:
        print(key, kwargs[key])
        setattr(conf, key, kwargs[key])
    return conf

def tile_images(imgs, picturesPerRow=4):
    """ Code borrowed from
    https://stackoverflow.com/questions/26521365/cleanly-tile-numpy-array-of-images-stored-in-a-flattened-1d-format/26521997
    """

    # Padding
    if imgs.shape[0] % picturesPerRow == 0:
        rowPadding = 0
    else:
        rowPadding = picturesPerRow - imgs.shape[0] % picturesPerRow
    if rowPadding > 0:
        imgs = np.concatenate([imgs, np.zeros((rowPadding, *imgs.shape[1:]), dtype=imgs.dtype)], axis=0)

    # Tiling Loop (The conditionals are not necessary anymore)
    tiled = []
    for i in range(0, imgs.shape[0], picturesPerRow):
        tiled.append(np.concatenate([imgs[j] for j in range(i, i + picturesPerRow)], axis=1))

    tiled = np.concatenate(tiled, axis=0)
    return tiled


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True, tile=False, picturesPerRow=4):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if image_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(image_tensor.size(0)):
            one_image = image_tensor[b]
            one_image_np = tensor2im(one_image)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np, picturesPerRow=picturesPerRow)
            return images_tiled
        else:
            return images_np

    if image_tensor.dim() == 2:
        image_tensor = image_tensor.unsqueeze(0)
    image_numpy = image_tensor.detach().cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


# Converts a one-hot tensor into a colorful label map
def tensor2label(label_tensor, n_label, imtype=np.uint8, tile=False):
    if label_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(label_tensor.size(0)):
            one_image = label_tensor[b]
            one_image_np = tensor2label(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if label_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(label_tensor, imtype)
    label_tensor = label_tensor.cpu().float()
    if label_tensor.size()[0] > 1:
        label_tensor = label_tensor.max(0, keepdim=True)[1]
    label_tensor = Colorize(n_label)(label_tensor)
    label_numpy = np.transpose(label_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result

def tensor2shapeim(shape_tensor, input_semantics):
    t = shape_tensor
    bs, nc, h, w = t.shape
    label_set = (input_semantics.sum(dim=(2, 3)) > 0).float().cpu()
    t_out = []
    blank = np.zeros((h, w), dtype=np.uint8)
    for b in range(label_set.shape[0]):
        shapes = []
        for c in range(label_set.shape[1]):
            if label_set[b, c].item() != 0:
                #s = util.tensor2im(t[b, c]) # gray image
                s = tensor2im(t[b, c]) # gray image
                shapes.append(s)
            else: # append dark image
                shapes.append(blank)

        shapes = np.hstack(shapes) # stack by row
        t_out.append(shapes)
    t = np.vstack(t_out)
    return t

def tensor2colorshapeim(shape_tensor, n_label, imtype=np.uint8, tile=False):
    t = shape_tensor

    if shape_tensor.dim() == 4:
        # transform each image in the batch
        images_np = []
        for b in range(shape_tensor.size(0)):
            one_image = shape_tensor[b]
            one_image_np = tensor2colorshapeim(one_image, n_label, imtype)
            images_np.append(one_image_np.reshape(1, *one_image_np.shape))
        images_np = np.concatenate(images_np, axis=0)
        if tile:
            images_tiled = tile_images(images_np)
            return images_tiled
        else:
            images_np = images_np[0]
            return images_np

    if shape_tensor.dim() == 1:
        return np.zeros((64, 64, 3), dtype=np.uint8)
    if n_label == 0:
        return tensor2im(shape_tensor, imtype)
    shape_tensor = shape_tensor.cpu().float()
    if shape_tensor.size()[0] > 1:
        shape_tensor = (shape_tensor)
    shape_tensor = Colorize(n_label)(shape_tensor)
    label_numpy = np.transpose(shape_tensor.numpy(), (1, 2, 0))
    result = label_numpy.astype(imtype)
    return result


def save_image(image_numpy, image_path, create_dir=False):
    if create_dir:
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
    if len(image_numpy.shape) == 2:
        image_numpy = np.expand_dims(image_numpy, axis=2)
    if image_numpy.shape[2] == 1:
        image_numpy = np.repeat(image_numpy, 3, 2)
    image_pil = Image.fromarray(image_numpy)

    # save to png
    image_pil.save(image_path.replace('.jpg', '.png'))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]


def natural_sort(items):
    items.sort(key=natural_keys)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def find_class_in_module(target_cls_name, module):
    target_cls_name = target_cls_name.replace('_', '').lower()
    clslib = importlib.import_module(module)
    cls = None
    for name, clsobj in clslib.__dict__.items():
        if name.lower() == target_cls_name:
            cls = clsobj

    if cls is None:
        print("In %s, there should be a class whose name matches %s in lowercase without underscore(_)" % (module, target_cls_name))
        exit(0)

    return cls


def save_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_path = os.path.join(opt.checkpoints_dir, opt.name, save_filename)
    torch.save(net.cpu().state_dict(), save_path)
    if len(opt.gpu_ids) and torch.cuda.is_available():
        net.cuda()


def load_network(net, label, epoch, opt):
    save_filename = '%s_net_%s.pth' % (epoch, label)
    save_dir = os.path.join(opt.checkpoints_dir, opt.name)
    save_path = os.path.join(save_dir, save_filename)
    weights = torch.load(save_path)
    net.load_state_dict(weights)
    return net


###############################################################################
# Code from
# https://github.com/ycszen/pytorch-seg/blob/master/transform.py
# Modified so it complies with the Citscape label map colors
###############################################################################
def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])


def labelcolormap(N):
    if N == (18 + 2): # human parsing
        cmap = np.array([(111, 111, 111), # 0 bg
                         (134, 69, 0),    # 1 hat
                         (180, 0, 0),       # 2 hair
                         (0, 0, 0), # 3 sunglass
                         (145, 134, 176),    # 4 upper clothes
                         (253, 215, 228),     # 5 skirt
                         (60, 156, 246),    # 6 pants
                         (253, 107, 190),  # 7 dress
                         (245, 134, 20), # 8 belt
                         (150, 75, 0),  # 9 left-shoe
                         (0, 150, 75), # 10 right-shoe
                         (255, 219, 172),  # 11 face
                         (0, 211, 211), # 12 left-leg
                         (211, 211, 0),  # 13 right-leg
                         (80, 200, 120), # 14 left-arm
                         (188, 156, 0),  # 15 right-arm
                         (0, 0, 230), # 16 bag
                         (192, 209, 213),  # 17 scarf
                         (111, 111, 111), # nothing
                         (111, 111, 111)],  # nothing
                        dtype=np.uint8)
    elif N == (18 + 1 + 2): # celeba
        cmap = np.array([
            (111, 111, 111), # 0
            (128,   0,   0), # 1
            (  0, 128,   0), # 2 green, on paper
            (128, 128,   0), # 3
            (  0,   0, 128), # 4
            (128,   0, 128), # 5
            (  0, 128, 128), # 6
            (128, 128, 128), # 7
            ( 64,   0,   0), # 8
            (192,   0,   0), # 9
              (200, 161, 9), # 10, r_eye to gold
            (192, 128,   0), # 11
            ( 64,   0, 128), # 12
            (192,   0, 128), # 13
            ( 64, 128, 128), # 14
            (192, 128, 128), # 15
            (  0,  64,   0), # 16
            (128,  64,   0), # 17
            (  0, 192,   0), # 18
            (128, 192,   0), # 19 no use
            (  0,  64, 128)], # 20 no use
            dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        cmap[0, :] = 111
        
        for i in range(1, N):
            r, g, b = 0, 0, 0
            id = i + 0
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7 - j))
                g = g ^ (np.uint8(str_id[-2]) << (7 - j))
                b = b ^ (np.uint8(str_id[-3]) << (7 - j))
                id = id >> 3
            
            cmap[i, 0] = r
            cmap[i, 1] = g
            cmap[i, 2] = b

    return cmap


class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image

# added by yenchi
def is_inf(tensor):
    if (tensor == float('inf')).any() or (tensor == float('-inf')).any():
        return True
    else:
        return False

def print_loss_dict(loss):
    msg = ""
    for k, v in loss.items():
        msg += "%s: %.4f | " % (k, v)

    return msg[:-3]

def get_data_generator(loader):
    while True:
        for data in loader:
            yield data

# save pixel-annotation as .png
def save_ann(ann, name='name.png'):
	imageio.imwrite(name, (ann / ann.max() * 255.).astype(np.uint8))

label_nc_dict = {
               'I32': 1000, 'I32_hdf5': 1000,
               'I64': 1000, 'I64_hdf5': 1000,
               'I128': 1000, 'I128_hdf5': 1000,
               'I256': 1000, 'I256_hdf5': 1000,
               'C10': 10, 'C100': 100,
               }
               
def get_fg_mask(seg, threshold=0.4):
    return (seg.sum(dim=(1,)) > threshold).float()

def set_bg_value(fg, threshold=0.4):
    fg_mask = (fg.sum(dim=(1,)) > threshold).float()

    bg = 1-fg_mask
    fg[:, 0] = bg

    return fg

def check_inf_nan(tensor):
    return (tensor == float('-inf')).any().item() == 1 or \
           (tensor == float('inf')).any().item() == 1 or \
           (torch.isnan(tensor).any().item() == 1)