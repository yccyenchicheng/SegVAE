import os
import sys
import pickle
import argparse
from datetime import datetime

import torch
import dataset
import models

from models import networks

from util import util

current_time = datetime.now().strftime("%m%d-%H%M%S")

def setup_expdir(opt):
    config_dict = {
        'gpu': opt.gpu_ids,
        'res': opt.size,
        'dset': opt.dataset,
        'bs': opt.batch_size,
        'lr': opt.lr,

        # Enc
        'shape': opt.shape_dim,
        'hidden': opt.hidden_dim,
        'z': opt.z_dim,

        # loss weights
        'lda_kl': opt.lambda_kl, 
        'sn': opt.use_sn,
    }
    # setup exp dir
    name = ''
    if opt.continue_train:
        name = 'continue_'

    log_root = opt.log_root
    for k, v in config_dict.items():
        name += "{}-{}_".format(k, v)
    name = name[:-1]
    opt.name = name

    opt.exp_dir = os.path.join(log_root, opt.name)
    util.mkdir(opt.exp_dir)

    # write config
    exp_dir = opt.exp_dir

    msg = ''
    if opt.continue_train:
        msg += '=> continue from: %s' % opt.ckpt_path
    msg += '\t=> Config:\n'
        
    for key, val in config_dict.items():
        msg += '\t\t{}:   {}\n'.format(key, val)

    print(msg)
    with open("%s/config.txt" % exp_dir, 'w') as config_file:
        config_file.write(msg)

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        ## base options
        parser.add_argument('--name', type=str, default='exp%s-' % current_time, help="name of this experiment (default: %(default)s)")
        parser.add_argument('--gpu_ids', type=str, default='0', help="0,1,2 for GPU 0, 1, 2")

        # continue training
        parser.add_argument('--continue_train', action='store_true', default=False, help='continue training or not')
        parser.add_argument('--ckpt_path', type=str, default='', help='ckpt to load')
        parser.add_argument('--opt_path', type=str, default='',
                            help='opt to load. if not provided, then it is: [ckpt_path]_opt.pkl')

        # input/output sizes
        parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='if the label map contains dontcare label (dontcare=255)')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        ## setting input
        # opt.dataroot will be: data/opt.dataset
        parser.add_argument('--dataset', type=str, default='celebamaskhq', choices=['humanparsing', 'celebamaskhq'])
        parser.add_argument('--size', type=int, default=128, help="it equals the resolution")
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--load_from_opt_file', action='store_true', help='load the options from checkpoints and use that as default')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--nThreads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--label_len', type=int, default=-1, help='label-set len to generate. -1: default setting, do not specify any length')

        # for instance-wise features
        parser.add_argument('--no_instance', default=1, help='if 1, do *not* add instance map as input')

        ## spectral norm stuff
        parser.add_argument('--SN_eps', type=float, default=1e-8, help='epsilon value to use for Spectral Norm(default: %(default)s)')
        parser.add_argument('--SN_num_G_SVs', type=int, default=1, help='Number of SVs to track in G (default: %(default)s)')
        parser.add_argument('--SN_num_D_SVs', type=int, default=1, help='Number of SVs to track in D (default: %(default)s)')
        parser.add_argument('--SN_num_G_SV_itrs', type=int, default=1, help='Number of SV itrs in G (default: %(default)s)')
        parser.add_argument('--SN_num_D_SV_itrs', type=int, default=1, help='Number of SV itrs in D (default: %(default)s)')

        ## train options
        # for displays
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')

        # for training
        parser.add_argument('--niter', type=int, default=150, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=50, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.999, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=2.5e-5, help='initial learning rate for adam.')

        parser.add_argument('--use_sn', type=int, default=1, choices=[0, 1], help='use spectral norm or not')

        ## misc
        # for convenience
        parser.add_argument('--current_time', default=current_time)

        # visualization
        parser.add_argument('--plot_freq', type=int, default=200)

        # for debug
        parser.add_argument('--test', action='store_true')
        parser.add_argument('--debug', type=int, default=0, choices=[0, 1])

        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify dataset-related parser options
        dataset_mode = opt.dataset
        dataset_option_setter = dataset.get_option_setter(dataset_mode, opt=opt) 
        parser = dataset_option_setter(parser)

        opt, unknown = parser.parse_known_args()

        # also make sure that ckpt_path is provided if it's in testing
        if not self.isTrain:
            assert opt.ckpt_path != '', 'should provide ckpt_path in testing mode'

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file or not self.isTrain:
            #assert opt.opt_path != '', 'should provide opt_path'
            if opt.opt_path == '':
                opt.opt_path = '%s_opt.pkl' % (opt.ckpt_path.replace('.pth', ''))

            parser = self.update_options_from_file(parser, opt, opt.opt_path)
            
        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.log_root, opt.name)
        
        if not os.path.exists(expr_dir):
            util.mkdirs(expr_dir)

        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt, opt_path=None):
        if opt_path is None:
            new_opt = self.load_options(opt)
        else:
            new_opt = pickle.load(open(opt_path, 'rb'))

        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})

        if not self.isTrain:
            # if k is not in opt, update
            for k, v in sorted(vars(new_opt).items()):
                if not hasattr(opt, k):
                    new_val = getattr(new_opt, k)
                    parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options() # opt_file will be loaded here
        opt.isTrain = self.isTrain   # train or test

        ## setup paths
        opt.dataroot = 'data/segvae/%s' % opt.dataset
        log_root = 'logs/segvae_logs'
        opt.log_root = log_root

        if not os.path.exists(log_root):
            os.mkdir(log_root)

        opt.total_epochs = opt.niter + opt.niter_decay
        
        # setup logs dir and ckpt_dir. (for training)
        if not opt.load_from_opt_file and opt.isTrain:
            setup_expdir(opt)
            opt.ckpt_dir = os.path.join(log_root, opt.name, 'checkpoint')
            util.mkdir(opt.ckpt_dir)
        else:
            opt.isTrain = False

        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0)
        if opt.no_instance != 1:
            opt.semantic_nc += 1

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batch_size % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batch_size, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
