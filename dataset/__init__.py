import importlib
import torch.utils.data
from torch.utils.data.dataloader import default_collate

def get_dataset_class(opt):

    dataset_filename = "dataset.%s_dataset" % opt.dataset
    datasetlib = importlib.import_module(dataset_filename)

    if opt.dataset == 'celebamaskhq':
        dataset_class = datasetlib.__dict__['CelebAMaskDatasetHQ']
    elif opt.dataset == 'humanparsing':
        dataset_class = datasetlib.__dict__['HumanParsingDataset']
    else:
        raise ValueError('|dataset| invalid')

    return dataset_class

def get_option_setter(dataset_name, opt=None):

    dataset_class = get_dataset_class(opt)

    return dataset_class.modify_commandline_options

def create_dataloader(opt, phase):

    collate_fn = default_collate

    dataset_class = get_dataset_class(opt)
    instance = dataset_class()
    instance.initialize(opt, phase, opt.label_len) # NOTE: label_len: for rebuttal. normal training can ignore this

    print("=> [%s] dataset [%s] of size %d was created" %
          (phase, type(instance).__name__, len(instance)))

    drop_last = True if phase == 'train' else False

    dataloader = torch.utils.data.DataLoader(
        instance,
        batch_size=opt.batch_size,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=drop_last,
        collate_fn=collate_fn,
    )

    return dataloader
