import argparse
import os
import sys
import random
from collections import OrderedDict
from datetime import datetime

import numpy as np
from tqdm import tqdm
from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import dataset
import util.util as util
from util.visualizer import Visualizer
from options.train_options import TrainOptions
from trainers.segvae_trainer import SegVAETrainer

# Setup cudnn.benchmark for free speed
torch.backends.cudnn.benchmark = True

with open('latest_cmd.txt', 'w') as f:
    cmd = ' '.join(sys.argv) + '\n'
    print(cmd)
    f.write(cmd)

opt = TrainOptions().parse()
config = vars(opt)

# setup seed
print("=> random seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
np.random.seed(opt.seed)

# setup data
train_dataloader = dataset.create_dataloader(opt, phase='train')
test_dataloader = dataset.create_dataloader(opt, phase='test')
test_batch_generator = util.get_data_generator(test_dataloader)

# setup trainer and model
trainer = SegVAETrainer(opt)
if not opt.continue_train:
    start_epoch = 0
else:
    start_epoch = opt.start_epoch + 1
    
# create tool for visualization
visualizer = Visualizer(opt)

total_steps_so_far = 0

for epoch in range(start_epoch, start_epoch + opt.total_epochs):

    for i, data_i in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        trainer.seg_vae_model.train()
        current_step = epoch*len(train_dataloader) + i

        ## Training
        trainer.run_one_step(data_i)

        if current_step % 1 == 0:
            if i != 0:
                sys.stdout.write("\033[F") #back to previous line
                sys.stdout.write("\033[K") #clear line

            loss_str = trainer.get_loss_str()
            print("[\033[91m%s SegVAE\033[00m][%d/%d] %d %s" % (opt.current_time, epoch, start_epoch + opt.total_epochs, current_step, loss_str))
            with open(visualizer.log_name, "a") as log_file:
                log_file.write('%s\n' % loss_str)
            
            # write to tensorboard
            trainer.log_tensorboard(data_i, current_step, 'train', mode='loss')

        if current_step % opt.plot_freq == 0:
            visuals = trainer.get_visuals(data_i)
            visualizer.display_current_results(visuals, epoch, current_step, 'train')
                
            trainer.log_tensorboard(data_i, current_step, phase='train', mode='visualization')

        # testing
        if current_step % opt.plot_freq == 0:
            # evaluation
            trainer.seg_vae_model.eval()
            test_data_i = next(test_batch_generator)
            visuals = trainer.run_inference(test_data_i)
            visualizer.display_current_results(visuals, epoch, current_step, 'test')
            trainer.log_tensorboard(test_data_i, current_step, phase='test', mode='visualization')

        if opt.test and (i + 1) % 5 == 0:
            break

    trainer.update_learning_rate(epoch)
    if epoch % 10 == 0 and opt.max_dataset_size > 20:
        model_path = os.path.join(opt.ckpt_dir, 'model_%d.pth' % epoch)
        trainer.save(model_path, epoch)
    model_path = os.path.join(opt.ckpt_dir, 'model_latest.pth')
    trainer.save(model_path, epoch)

print('Training complete.')
