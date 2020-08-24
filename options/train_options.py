import os
import sys
import argparse

from datetime import datetime
import dataset

from util import util
from .base_options import BaseOptions

class TrainOptions(BaseOptions):

    def initialize(self, parser):
        BaseOptions.initialize(self, parser)

        # seed
        parser.add_argument('--seed', type=int, default=111, help='seed for reproducibility')

        # for encoder, decoder
        parser.add_argument('--ngf', type=int, default=64, help='how many base filters for G (encoder and decoder)')

        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='variance of the initialization distribution')

        parser.add_argument('--enc_norm', default='in', choices=['bn', 'in', 'none'], help='encoder: batchnorm or instance norm')
        parser.add_argument('--dec_norm', default='in', choices=['bn', 'in', 'none'], help='decoder: batchnorm or instance norm')

        # latent space
        parser.add_argument('--shape_dim', type=int, default=512, help="dimension of the encoded shape")
        parser.add_argument('--hidden_dim', type=int, default=256, help="dimension of the hidden_dim")

        parser.add_argument('--z_dim', type=int, default=384, help="dimension of the latent z vector")

        parser.add_argument('--lambda_kl', type=float, default=1e-5, help='weight for kl loss')
        parser.add_argument('--lambda_shape', type=float, default=1, help='weight for shape loss')

        self.isTrain = True
        return parser