import random

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import torchvision.transforms as transforms
from torch.nn import init

import models.networks as networks
from models.networks.encoder import ShapeEncoder, PriorEncoder, GaussianEncoder
from models.networks.decoder import ShapeDecoder
from models.networks.loss import KLDLoss, KLDLossNoReduction, KLDLossWithStandardGaussianNoReduction

import util
from util.util import is_inf
from util import labels

order_dict = {'celebamaskhq': labels.celeba_order, 'humanparsing': labels.humanparsing_order}

def normalize(tensor):
    # assum a batch of img tensor, range: 0~1
    return (tensor - 0.5)/0.5

class SegVAEModel(torch.nn.Module):
    """ SegVAE model. """

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor

        # create_networks
        self.shapeE = ShapeEncoder(opt)
        self.prior = PriorEncoder(opt)
        self.posterior = GaussianEncoder(opt)
        self.shapeD = ShapeDecoder(opt)

        if len(opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.shapeE.cuda()
            self.prior.cuda()
            self.posterior.cuda()
            self.shapeD.cuda()

        self.shapeE.init_weights(opt.init_type, opt.init_variance)
        self.prior.init_weights(opt.init_type, opt.init_variance)
        self.posterior.init_weights(opt.init_type, opt.init_variance)
        self.shapeD.init_weights(opt.init_type, opt.init_variance)

        if self.opt.continue_train or not self.opt.isTrain:
            self.shapeE, self.shapeD, self.prior, self.posterior, epoch = self.load(self.opt.ckpt_path)
            print("=> load model from: %s, start_epoch: %d" % (self.opt.ckpt_path, epoch))
            opt.start_epoch = epoch

        print("=> finish initializing model")

        ## loss
        if hasattr(self, 'prior'):
            self.criterion_KL = KLDLossNoReduction()
        else:
            self.criterion_KL = KLDLossWithStandardGaussianNoReduction()

        self.criterion_recon_shape = nn.L1Loss(reduction='none')    # default reduction: mean

        self.order = order_dict[self.opt.dataset]
        if self.opt.dataset == 'celebamaskhq':
            order_str = [labels.celeba_idx_to_cls[i] for i in self.order]
        elif self.opt.dataset == 'humanparsing':
            order_str = [labels.humanparsing_idx_to_cls[i] for i in self.order]
        else:
            raise NotImplementedError

        print('=> order is {}'.format(order_str))

        self.normalize = normalize
        self.threshold = 0.4 # threshold for producing background
        
    def forward(self, data, mode):
        if self.opt.dataset == 'celebamaskhq':
            preprocess_method = self.preprocess_celeba_input
        elif self.opt.dataset == 'humanparsing':
            preprocess_method = self.preprocess_humanparsing_input
        else:
            raise NotImplementedError

        real_image, input_semantics, label_set = preprocess_method(data)

        if mode == 'train':
            losses, recon, gen_prior, input_semantics = self.compute_loss(real_image, input_semantics, label_set)
            return losses, recon, gen_prior, input_semantics
        elif mode == 'inference':
            with torch.no_grad():
                recon, gen_prior, input_semantics = self.inference(real_image, input_semantics, label_set)
            return recon, gen_prior, input_semantics
        else:
            raise ValueError("|%s| invalid" % mode)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu

    def create_optimizers(self, opt):
        shapeE_param = list(self.shapeE.parameters())
        shapeD_param = list(self.shapeD.parameters())
        prior_param = list(self.prior.parameters())
        posterior_param = list(self.posterior.parameters())

        lr = opt.lr
        beta1, beta2 = opt.beta1, opt.beta2

        optimizer_shapeE = optim.Adam(shapeE_param, lr=lr, betas=(beta1, beta2))
        optimizer_shapeD = optim.Adam(shapeD_param, lr=lr, betas=(beta1, beta2))
        optimizer_prior = optim.Adam(prior_param, lr=lr, betas=(beta1, beta2))
        optimizer_posterior = optim.Adam(posterior_param, lr=lr, betas=(beta1, beta2))

        return optimizer_shapeE, optimizer_shapeD, optimizer_prior, optimizer_posterior

    def save(self, path, epoch):
        states = {
            'shapeE': self.shapeE.cpu().state_dict(),
            'shapeD': self.shapeD.cpu().state_dict(),
            'prior': self.prior.cpu().state_dict(),
            'posterior': self.posterior.cpu().state_dict(),
            'epoch': epoch,
        }
        torch.save(states, path)
        if len(self.opt.gpu_ids) > 0:
            assert(torch.cuda.is_available())
            self.shapeE.cuda()
            self.shapeD.cuda()
            self.prior.cuda()
            self.posterior.cuda()

    def load(self, path):
        states = torch.load(path)
        self.shapeE.load_state_dict(states['shapeE'])
        self.shapeD.load_state_dict(states['shapeD'])
        self.prior.load_state_dict(states['prior'])
        self.posterior.load_state_dict(states['posterior'])
        epoch = states['epoch']
        return self.shapeE, self.shapeD, self.prior, self.posterior, epoch

    ##################################################################################

    def preprocess_celeba_input(self, data):
        # move to GPU and change data types
        if self.use_gpu():
            data['label'] = data['label'].cuda()

        image = 0

        input_semantics = data['label']
        bs, nc, h, w = input_semantics.size()

        assert input_semantics[:, 0].sum().item() == 0

        label_set = (input_semantics.view(bs, nc, -1).sum(-1) > 0).float()
        label_set[:, 0] = 0. # set bg to 0

        return image, input_semantics, label_set

    def preprocess_humanparsing_input(self, data):

        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label'].cuda()

        # normalize
        image = 0

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc

        input_label = self.FloatTensor(bs, nc, h, w).zero_()
        input_semantics = input_label.scatter_(1, label_map, 1.0)

        # NOTE: set bg as 0.
        input_semantics[:, 0] = 0.

        label_set = (input_semantics.view(bs, nc, -1).sum(-1) > 0).float()
        label_set[:, 0] = 0. # set bg to 0

        return image, input_semantics, label_set

    def compute_loss(self, real_image, input_semantics, label_set):
        losses = {}
        losses['KLD'] = 0
        losses['recon_shape'] = 0

        pred_order = self.order

        bs, nc, h, w = input_semantics.shape

        recon     = self.FloatTensor(bs, nc, h, w).zero_()
        gen_prior = self.FloatTensor(bs, nc, h, w).zero_()

        x_prior = self.FloatTensor(bs, nc, h, w).fill_(0)

        # initialize LSTM
        device = input_semantics.device
        self.prior.init_hidden(device=device, batch_size=bs)
        self.posterior.init_hidden(device=device, batch_size=bs)

        valid_cnt = 0
        for i in range(nc-1): # don't train background
            c = pred_order[i]
            x_current = input_semantics[:, c].unsqueeze(dim=1)
            if i == 0:
                x_prior = x_prior
            else:
                c_prev = pred_order[i-1]
                x_prior[:, c_prev] = input_semantics[:, c_prev]

            mask = (input_semantics[:, c].sum(dim=(1, 2)) > 0).float()
            valid_batch = mask.sum().long().item()

            # if valid_batch == 0:
            #     continue

            valid_cnt += 1
            
            y_c = self.FloatTensor(bs, nc).zero_()
            y_c[mask == 1, c] = 1

            # encode
            feat_prior = self.shapeE(x_prior, label_set, y_c)

            z_prior, mu_prior, logvar_prior = self.prior(feat_prior)
            z_post, mu_post, logvar_post = self.posterior(x_current, feat_prior, y_c)

            if valid_batch == 0:
                continue

            recon_shape_c = self.shapeD(torch.cat([feat_prior, z_post], dim=1)) * mask[..., None, None, None]

            with torch.no_grad():
                prior_gen_shape = self.shapeD(torch.cat([feat_prior.detach(), z_prior.detach()], dim=1)) * mask[..., None, None, None] 

            kld_loss = self.criterion_KL(mu_post, logvar_post, mu_prior, logvar_prior) * mask[..., None]
            kld_loss = kld_loss.sum() / valid_batch

            shape_loss = self.criterion_recon_shape(recon_shape_c, x_current) * mask[..., None, None, None]
            shape_loss = shape_loss.sum() / (valid_batch * h * w)

            losses['KLD'] += kld_loss * self.opt.lambda_kl
            losses['recon_shape'] += shape_loss * self.opt.lambda_shape

            recon[:, c] = recon_shape_c[:, 0].detach()
            gen_prior[:, c] = prior_gen_shape[:, 0].detach()

        # background mask
        recon = recon.clone().detach()
        gen_prior = gen_prior.clone().detach()
        threshold = self.threshold
        recon_mask = self.get_fg_mask(recon, threshold)
        prior_mask = self.get_fg_mask(gen_prior, threshold)

        recon_bg = 1-recon_mask
        prior_bg = 1-prior_mask
        recon[:, 0] = recon_bg
        gen_prior[:, 0] = prior_bg

        return losses, recon, gen_prior, input_semantics

    def inference(self, real_image, input_semantics, label_set):
        recon, gen_prior = self.generate_fake(real_image, input_semantics, label_set)
        return recon, gen_prior, input_semantics

    def generate_fake(self, real_image, input_semantics, label_set):
        
        pred_order = self.order

        bs, nc, h, w = input_semantics.shape

        recon = self.FloatTensor(bs, nc, h, w).zero_()
        gen_prior = self.FloatTensor(bs, nc, h, w).zero_()

        x_prior = self.FloatTensor(bs, nc, h, w).fill_(0)

        # initialize LSTM
        device = input_semantics.device
        self.prior.init_hidden(device=device, batch_size=bs)
        self.posterior.init_hidden(device=device, batch_size=bs)

        for i in range(nc-1):
            c = pred_order[i]
            x_current = input_semantics[:, c].unsqueeze(dim=1)
            if i == 0:
                x_prior = x_prior
            else:
                c_prev = pred_order[i-1]
                x_prior[:, c_prev] = gen_prior[:, c_prev]

            mask = (input_semantics[:, c].sum(dim=(1, 2)) > 0).float()
            valid_batch = mask.sum()

            y_c = self.FloatTensor(bs, nc).zero_()
            y_c[mask == 1, c] = 1

            feat_prior = self.shapeE(x_prior, label_set, y_c)
            z_prior, mu_prior, logvar_prior = self.prior(feat_prior)
            z_post, mu_post, logvar_post = self.posterior(x_current, feat_prior, y_c)

            if valid_batch.item() == 0:
                continue

            recon_shape_c = self.shapeD(torch.cat([feat_prior, z_post], dim=1)) * mask[..., None, None, None]
            prior_gen_shape = self.shapeD(torch.cat([feat_prior, z_prior], dim=1)) * mask[..., None, None, None] 

            recon[:, c] = recon_shape_c[:, 0].detach()
            gen_prior[:, c] = prior_gen_shape[:, 0].detach()
        

        # mask out bg
        recon = recon.clone().detach()
        gen_prior = gen_prior.clone().detach()

        # background mask
        threshold = self.threshold
        recon_mask = self.get_fg_mask(recon, threshold)
        prior_mask = self.get_fg_mask(gen_prior, threshold)
        
        recon_bg = 1-recon_mask
        prior_bg = 1-prior_mask
        recon[:, 0] = recon_bg
        gen_prior[:, 0] = prior_bg

        return recon, gen_prior
    
    def generate_from_label_set(self, label_set):
        pred_order = self.order
        
        bs, nc = label_set.shape
        h, w = self.opt.size, self.opt.size

        gen_prior = self.FloatTensor(bs, nc, h, w).zero_()

        x_prior = self.FloatTensor(bs, nc, h, w).fill_(0)

        # initialize LSTM
        device = label_set.device
        self.prior.init_hidden(device=device, batch_size=bs)

        for i in range(nc-1):
            c = pred_order[i]
            if i == 0:
                x_prior = x_prior
            else:
                c_prev = pred_order[i-1]
                x_prior[:, c_prev] = gen_prior[:, c_prev]

            mask = (label_set[:, c] > 0).float()
            valid_batch = mask.sum()

            y_c = self.FloatTensor(bs, nc).zero_()
            y_c[mask == 1, c] = 1

            feat_prior = self.shapeE(x_prior, label_set, y_c)
            z_prior, mu_prior, logvar_prior = self.prior(feat_prior)

            if valid_batch.item() == 0:
                continue

            prior_gen_shape = self.shapeD(torch.cat([feat_prior, z_prior], dim=1)) * mask[..., None, None, None] 
            gen_prior[:, c] = prior_gen_shape[:, 0].detach()
        
        # mask out bg
        gen_prior = gen_prior.clone().detach()

        # background mask
        threshold = self.threshold
        prior_mask = self.get_fg_mask(gen_prior, threshold)        
        prior_bg = 1-prior_mask
        gen_prior[:, 0] = prior_bg

        return gen_prior

    def get_fg_mask(self, fg, threshold=0.5):
        fg_mask = (fg.sum(dim=(1,)) > threshold).float()
        return fg_mask

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

