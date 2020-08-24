import os
import ntpath
import time
from . import util
from . import html
import numpy as np
import scipy.misc
import imageio
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

class Visualizer():
    def __init__(self, opt):
        self.opt = opt
        self.tf_log = opt.isTrain and opt.tf_log
        self.use_html = not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        if self.use_html:
            self.web_dir = os.path.join(opt.log_root, opt.name, 'web')
            self.web_img_dir = os.path.join(self.web_dir, 'images')
            util.mkdirs([self.web_dir, self.web_img_dir])

        # training mode
        if self.opt.isTrain:
            opt.image_dir = os.path.join(opt.log_root, opt.name, 'images')
            self.img_dir = opt.image_dir
            util.mkdirs([os.path.join(self.img_dir, 'train'), os.path.join(self.img_dir, 'test')])
            print('=> creating images directory %s...' % self.img_dir)
        else:
            # testing mode
            self.img_dir = opt.image_dir # --> results_dir in testing mode

        self.log_name = os.path.join(opt.log_root, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, step, phase):

        ## convert tensors to numpy arrays
        visuals = self.convert_visuals_to_numpy(visuals)

        # write to disk
        for label, image_numpy in visuals.items():
            if isinstance(image_numpy, list):
                for i in range(len(image_numpy)):
                    img_path = os.path.join(self.img_dir, phase, 'epoch%.3d_iter%.3d_%s_%d.png' % (epoch, step, label, i))
                    util.save_image(image_numpy[i], img_path)
            else:
                try:
                    img_path = os.path.join(self.img_dir, phase, 'epoch%.3d_iter%.3d_%s.png' % (epoch, step, label))
                    if len(image_numpy.shape) >= 4:
                        image_numpy = image_numpy[0]                    
                    util.save_image(image_numpy, img_path)
                except:
                    print('=> bug in visualizer.py. label: %s' % label)
                    import pdb; pdb.set_trace()

    # errors: dictionary of error labels and values
    def plot_current_errors(self, errors, step):
        if self.tf_log:
            for tag, value in errors.items():
                value = value.mean().float()
                summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=tag, simple_value=value)])
                self.writer.add_summary(summary, step)

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t):
        message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            v = v.mean().float()
            message += '%s: %.3f ' % (k, v)

        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def convert_visuals_to_numpy(self, visuals):
        for key, t in visuals.items():
            if 'input_semantics' in key:
                input_semantics = t

            tile = True
            if 'shape' in key:
                t = util.tensor2shapeim(t, input_semantics)
            elif 'label_map' in key:
                if self.opt.dataset == 'celebamaskhq':
                    assert t.shape[1] == self.opt.label_nc, '# of channels not right. required: label_nc: %d; t.shape[1]: %d' % (self.opt.label_nc, t.shape[1])
                else:
                    assert t.shape[1] == 1, '# of channels not right. required: 1. t.shape[1]: %d' % (t.shape[1])
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            elif 'seg' in key:
                assert t.shape[1] == self.opt.label_nc, '# of channels not right. label_nc: %d; t.shape[1]: %d' % (self.opt.label_nc, t.shape[1])
                t = util.tensor2label(t, self.opt.label_nc + 2, tile=tile)
            elif 'img' in key:
                assert t.shape[1] == 3, '# of channels not right. required: 3, t.shape[1]: %d' % (t.shape[1])
                t = util.tensor2im(t, tile=tile)

            visuals[key] = t
        return visuals

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):        
        visuals = self.convert_visuals_to_numpy(visuals)        
        
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = os.path.join(label, '%s.png' % (name))
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path, create_dir=True)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)