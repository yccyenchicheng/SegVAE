import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks import layers
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer
from models.networks.architecture import ResnetBlock

class ShapeEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))

        nf = opt.ngf        
        self.shape_dim = opt.shape_dim
        self.hidden_dim = self.opt.hidden_dim

        self.res = opt.size
        
        # define norm layer
        norm_layer = opt.enc_norm
        if norm_layer == 'in':
            norm_layer = nn.InstanceNorm2d
        elif norm_layer == 'bn':
            norm_layer = nn.BatchNorm2d

        # define conv layer and linear layer
        if opt.use_sn == 1:
            print('=> using spectral norm')
            conv_layer = functools.partial(layers.SNConv2d, kernel_size=kw,
                                           num_svs=opt.SN_num_G_SVs, num_itrs=opt.SN_num_G_SV_itrs,
                                           eps=opt.SN_eps)
            linear_layer = functools.partial(layers.SNLinear,
                                             num_svs=opt.SN_num_G_SVs, num_itrs=opt.SN_num_G_SV_itrs,
                                             eps=opt.SN_eps)
        else:
            conv_layer = functools.partial(nn.Conv2d, kernel_size=kw)
            linear_layer = nn.Linear

        # define activation
        actvn = nn.LeakyReLU(0.2, False)

        self.embed = nn.Sequential(
            linear_layer(opt.label_nc, self.hidden_dim),
            nn.ReLU(False),
            linear_layer(self.hidden_dim, self.res * self.res),
            nn.ReLU(False),
        )

        self.embed_label_set = nn.Sequential(
            linear_layer(opt.label_nc, self.hidden_dim),
            nn.ReLU(False),
            linear_layer(self.hidden_dim, self.res * self.res),
            nn.ReLU(False),
        )

        self.layer1 = ConvDownBlock(inc=opt.label_nc+1+1, outc=nf * 1, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer2 = ConvDownBlock(inc=nf * 1, outc=nf * 2, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer3 = ConvDownBlock(inc=nf * 2, outc=nf * 4, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer4 = ConvDownBlock(inc=nf * 4, outc=nf * 8, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer5 = ConvDownBlock(inc=nf * 8, outc=nf * 8, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)

        self.so = s0 = 4
        if opt.use_sn == 1:
            self.layer_out = layers.SNConv2d(nf * 8, self.shape_dim, kernel_size=s0, stride=1, padding=0, 
                                             num_svs=opt.SN_num_G_SVs, num_itrs=opt.SN_num_G_SV_itrs,
                                             eps=opt.SN_eps)
        else:
            self.layer_out = nn.Conv2d(nf * 8, self.shape_dim, kernel_size=s0, stride=1, padding=0)

        self.opt = opt

    def forward(self, x, label_set, y_c):
        if x.size(2) != self.res or x.size(3) != self.res:
            x = F.interpolate(x, size=(self.opt.res, self.opt.res), mode='bilinear')

        bs = x.shape[0]
        label_set = self.embed_label_set(label_set)
        y_c = self.embed(y_c)

        label_set = label_set.view(bs, 1, self.res, self.res)
        y_c = y_c.view(bs, 1, self.res, self.res)

        x = self.layer1(torch.cat([x, label_set, y_c], dim=1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.layer_out(x)
        x = x.view(x.size(0), -1)
        return x

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

class GaussianEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_layers = 1 # for lstm

        self.shape_dim = opt.shape_dim

        self.hidden_dim = self.opt.hidden_dim

        self.z_dim = opt.z_dim
        self.res = opt.size
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        nf = opt.ngf        
        
        # define norm layer
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

        # define activation
        actvn = nn.LeakyReLU(0.2, False)

        self.embed = nn.Sequential(
            nn.Linear(opt.label_nc, self.hidden_dim),
            nn.ReLU(False),
            nn.Linear(self.hidden_dim, self.res * self.res),
            nn.ReLU(False),
        )

        self.layer1 = ConvDownBlock(   inc=1+1, outc=nf * 1, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer2 = ConvDownBlock(inc=nf * 1, outc=nf * 2, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer3 = ConvDownBlock(inc=nf * 2, outc=nf * 4, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer4 = ConvDownBlock(inc=nf * 4, outc=nf * 8, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)
        self.layer5 = ConvDownBlock(inc=nf * 8, outc=nf * 8, conv_layer=conv_layer, s=2, p=pw, norm_layer=norm_layer)

        self.so = s0 = 4

        if opt.use_sn == 1:
            self.final_conv = layers.SNConv2d(nf * 8, self.hidden_dim, kernel_size=s0, stride=1, padding=0, 
                                             #num_svs=opt.num_G_SVs, num_itrs=opt.num_G_SV_itrs,
                                             num_svs=opt.SN_num_G_SVs, num_itrs=opt.SN_num_G_SV_itrs,
                                             eps=opt.SN_eps)
        else:
            self.final_conv = nn.Conv2d(nf * 8, self.hidden_dim, kernel_size=s0, stride=1, padding=0)

        self.lstm = nn.ModuleList([nn.LSTMCell(self.hidden_dim + self.shape_dim, self.hidden_dim) for i in range(self.n_layers)])
        self.fc_mu = linear_layer(self.hidden_dim, self.z_dim)
        self.fc_var = linear_layer(self.hidden_dim, self.z_dim)

        self.opt = opt

    def forward(self, x, feat, y_c):
        if x.size(2) != self.res or x.size(3) != self.res:
            x = F.interpolate(x, size=(self.opt.res, self.opt.res), mode='bilinear')

        bs = x.shape[0]
        y_c = self.embed(y_c)
        y_c = y_c.view(bs, 1, self.res, self.res)

        x = self.layer1(torch.cat([x, y_c],dim=1))
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)

        x = self.final_conv(x)
        x = x.view(x.size(0), -1)

        h_in = torch.cat([x, feat], dim=1)
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        mu = self.fc_mu(h_in)
        logvar = self.fc_var(h_in)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def init_hidden(self, device=None, batch_size=None):
        assert device is not None and batch_size is not None

        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(batch_size, self.hidden_dim, requires_grad=True).to(device),
                           torch.zeros(batch_size, self.hidden_dim, requires_grad=True).to(device)))
        self.hidden = hidden

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

class PriorEncoder(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.n_layers = 1 # for lstm

        self.shape_dim = opt.shape_dim
        
        self.hidden_dim = self.opt.hidden_dim

        self.z_dim = opt.z_dim
        self.res = opt.size

        # define conv layer and linear layer
        if opt.use_sn == 1:
            linear_layer = functools.partial(layers.SNLinear,
                                             num_svs=opt.SN_num_G_SVs, num_itrs=opt.SN_num_G_SV_itrs,
                                             eps=opt.SN_eps)
        else:
            linear_layer = nn.Linear

        self.embed = linear_layer(self.shape_dim, self.hidden_dim)
        self.lstm = nn.ModuleList([nn.LSTMCell(self.hidden_dim, self.hidden_dim) for i in range(self.n_layers)])
        self.fc_mu = linear_layer(self.hidden_dim, self.z_dim)
        self.fc_var = linear_layer(self.hidden_dim, self.z_dim)

        self.opt = opt

    def forward(self, x):
        x = self.embed(x)

        h_in = x
        for i in range(self.n_layers):
            self.hidden[i] = self.lstm[i](h_in, self.hidden[i])
            h_in = self.hidden[i][0]

        mu = self.fc_mu(h_in)
        logvar = self.fc_var(h_in)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def init_hidden(self, device=None, batch_size=None):
        assert device is not None and batch_size is not None

        hidden = []
        for i in range(self.n_layers):
            hidden.append((torch.zeros(batch_size, self.hidden_dim, requires_grad=True).to(device),
                           torch.zeros(batch_size, self.hidden_dim, requires_grad=True).to(device)))
        self.hidden = hidden

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

class ConvDownBlock(nn.Module):
    def __init__(self, inc, outc, conv_layer=nn.Conv2d, s=2, p=1, norm_layer='none'):
        super().__init__()

        if norm_layer == 'none':
            self.net = nn.Sequential(
                conv_layer(inc, outc, stride=s, padding=p),
                nn.LeakyReLU(0.2, False),
            )
        else:
            self.net = nn.Sequential(
                conv_layer(inc, outc, stride=s, padding=p),
                norm_layer(outc),
                nn.LeakyReLU(0.2, False)
            )
    
    def forward(self, x):
        return self.net(x)
