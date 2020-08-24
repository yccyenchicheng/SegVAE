import os
import numpy as np
import torch
from collections import OrderedDict
from tensorboardX import SummaryWriter

import util
from models.networks.sync_batchnorm import DataParallelWithCallback
from models.segvae_model import SegVAEModel

class SegVAETrainer():

    def __init__(self, opt):
        self.opt = opt
        self.seg_vae_model = SegVAEModel(opt)
        if len(opt.gpu_ids) > 0:
            self.seg_vae_model = torch.nn.DataParallel(self.seg_vae_model, device_ids=opt.gpu_ids)
            self.seg_vae_model_on_one_gpu = self.seg_vae_model.module
        else:
            self.seg_vae_model_on_one_gpu = self.seg_vae_model

        self.generated = None
        
        self.optimizer_shapeE, self.optimizer_shapeD, self.optimizer_prior, self.optimizer_posterior = self.seg_vae_model_on_one_gpu.create_optimizers(opt)

        self.old_lr = opt.lr
        self.has_set_fix_data = False

        tboard_dir = os.path.join(opt.log_root, opt.name, 'tboard')
        self.writer = SummaryWriter(log_dir=tboard_dir)

    def run_one_step(self, data):
        self.optimizer_shapeE.zero_grad()
        self.optimizer_shapeD.zero_grad()
        self.optimizer_prior.zero_grad()
        self.optimizer_posterior.zero_grad()

        losses, recon, gen_prior, input_semantics = self.seg_vae_model(data, mode='train')
        loss = sum(losses.values()).mean()
        loss.backward()

        self.optimizer_shapeE.step()
        self.optimizer_shapeD.step()
        self.optimizer_prior.step()
        self.optimizer_posterior.step()

        self.losses = losses
        self.recon = recon
        self.gen_prior = gen_prior
        self.input_semantics = input_semantics

    def run_inference(self, test_data):
        with torch.no_grad():
            recon, gen_prior, input_semantics = self.seg_vae_model(test_data, mode='inference')

        self.recon = recon
        self.gen_prior = gen_prior
        self.input_semantics = input_semantics
    
        visuals = self.get_visuals(test_data)
        return visuals

    def get_latest_results(self):
        return self.recon, self.gen_prior, self.input_semantics
        
    def get_loss_str(self):
        def gather_str(name, errors):
            msg = '%s: ' % name
            for k, v in errors.items():
                v = v.mean().float()
                msg += '%s: %.10f ' % (k, v)
            msg += '| '
            return msg
        
        losses = self.losses
        loss_str = gather_str('VAE', losses)
        return loss_str

    def get_visuals(self, data):
        
        v = 'SegVAE'
        recon, gen_prior, input_semantics = self.get_latest_results()
        if self.opt.dataset == 'celebamaskhq':
            input_semantics[:, 1][input_semantics[:, 1] == 1] = 0.9 # skin will shadow other classes
        
        visuals = OrderedDict([
                            # ('1-label_map-input_%s' % v, data['label']),
                            ('2-seg-input_%s' % v, input_semantics),
                            ('3-seg-recon_%s' % v, recon),
                            ('4-seg-prior_%s' % v, gen_prior),
                            ])
        return visuals

    def log_tensorboard(self, data, step, phase, mode):
        writer = self.writer
        if mode == 'loss':
            losses = self.losses
            for key, val in losses.items():
                writer.add_scalar('Train/%s' % key, val, step)

        elif mode == 'visualization':
            tile = True
            recon, gen_prior, input_semantics = self.get_latest_results()

            # set bg 1 for visualization
            input_semantics = util.util.set_bg_value(input_semantics, threshold=0.4)
            if self.opt.dataset == 'celebamaskhq':
                input_semantics[:, 1][input_semantics[:, 1] == 1] = 0.9 # skin will shadow other classes

            input_semantics_np = util.util.tensor2label(input_semantics, self.opt.label_nc + 2, tile=tile)
            recon_seg_np = util.util.tensor2label(recon, self.opt.label_nc + 2, tile=tile)
            gen_seg_prior_np = util.util.tensor2label(gen_prior, self.opt.label_nc + 2, tile=tile)

            writer.add_image('%s/seg_input' % phase, input_semantics_np.transpose(2, 0, 1), step)
            writer.add_image('%s/seg_recon' % phase, recon_seg_np.transpose(2, 0, 1), step)
            writer.add_image('%s/seg_prior' % phase, gen_seg_prior_np.transpose(2, 0, 1), step)
        else:
            raise ValueError('|mode| invalid')

    def save(self, path, epoch):
        self.seg_vae_model_on_one_gpu.save(path, epoch)

    def update_learning_rate(self, epoch):
        if epoch > self.opt.niter:
            lrd = self.opt.lr / self.opt.niter_decay
            new_lr = self.old_lr - lrd
        else:
            new_lr = self.old_lr

        if new_lr != self.old_lr:
            for param_group in self.optimizer_shapeE.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimizer_shapeD.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimizer_prior.param_groups:
                param_group['lr'] = new_lr
            for param_group in self.optimizer_posterior.param_groups:
                param_group['lr'] = new_lr
            print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
            self.old_lr = new_lr
