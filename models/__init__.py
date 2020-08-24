import importlib
import torch


def find_model_using_v(v):
    modelname = 'segvae'
    model_filename = "models." + "%s_model" % (modelname)
    target_model_name = "SegVAEModel"
    modellib = importlib.import_module(model_filename)

    # In the file, the class called ModelNameModel() will
    # be instantiated. It has to be a subclass of torch.nn.Module,
    # and it is case-insensitive.

    model = None

    for name, cls in modellib.__dict__.items():
        if name.lower() == target_model_name.lower() and issubclass(cls, torch.nn.Module):
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of torch.nn.Module with class name that matches %s in lowercase." % (model_filename, target_model_name))
        exit(0)

    return model

def get_option_setter(model_name):
    model_class = spade_find_model_using_name(model_name)
    return model_class.modify_commandline_options


def create_model(opt):
    model = spade_find_model_using_name(opt.model)
    instance = model(opt)
    print("model [%s] was created" % (type(instance).__name__))

    return instance