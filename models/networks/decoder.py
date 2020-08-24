import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import layers
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock

class ShapeDecoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))

        nf = opt.ngf
        self.shape_dim = opt.shape_dim
        self.z_dim = opt.z_dim
        self.res = opt.size
        
        norm_layer = opt.enc_norm
        if norm_layer == 'in':
            norm_layer = nn.InstanceNorm2d
        elif norm_layer == 'bn':
            norm_layer = nn.BatchNorm2d

        # define conv layer and linear layer
        if opt.use_sn == 1:
            conv_layer = functools.partial(layers.SNConv2d, kernel_size=kw,
                                           num_svs=opt.SN_num_G_SVs, num_itrs=opt.SN_num_G_SV_itrs,
                                           eps=opt.SN_eps)
            linear_layer = functools.partial(layers.SNLinear,
                                             num_svs=opt.SN_num_G_SVs, num_itrs=opt.SN_num_G_SV_itrs,
                                             eps=opt.SN_eps)
        else:
            conv_layer = functools.partial(nn.Conv2d, kernel_size=kw)
            linear_layer = nn.Linear

        self.s0 = s0 = 4
        self.layer_in = nn.ConvTranspose2d(self.shape_dim+self.z_dim, nf * 8, kernel_size=s0, stride=1, padding=0)

        self.layer1 = ConvUpBlock(inc=nf * 8, outc=nf * 8, k=kw, s=2, p=pw, norm_layer=norm_layer)
        self.layer2 = ConvUpBlock(inc=nf * 8, outc=nf * 4, k=kw, s=2, p=pw, norm_layer=norm_layer)
        self.layer3 = ConvUpBlock(inc=nf * 4, outc=nf * 2, k=kw, s=2, p=pw, norm_layer=norm_layer)
        self.layer4 = ConvUpBlock(inc=nf * 2, outc=nf * 1, k=kw, s=2, p=pw, norm_layer=norm_layer)
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(nf * 1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

        self.opt = opt

    def forward(self, x):
        bs = x.shape[0]
        x = x.view(bs, -1, 1, 1)
        x = self.layer_in(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

class ConvUpBlock(nn.Module):
    def __init__(self, inc, outc, k=3, s=2, p=1, norm_layer='none'):
        super().__init__()

        if norm_layer == 'none':
            self.net = nn.Sequential(
                nn.ConvTranspose2d(inc, outc, kernel_size=k, stride=s, padding=p, output_padding=1),
                nn.LeakyReLU(0.2, False)
            )
        else:
            self.net = nn.Sequential(
                nn.ConvTranspose2d(inc, outc, kernel_size=k, stride=s, padding=p, output_padding=1),
                norm_layer(outc),
                nn.LeakyReLU(0.2, False)
            )
    
    def forward(self, x):
        return self.net(x)