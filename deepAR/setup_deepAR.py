import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .deepAR import *
from .get_data_AR import *


def setup_deepAR(cfg, device, run, stimuli_index):
    cfg['data']["test_run"] = run
    data_params = cfg['data']
    
    window_size  = data_params['window_size']
    time_resolution = data_params['time_resolution']
    data_params['step_size'] = data_params['step_size'] if 'step_size' in data_params else 1
    step_size = data_params['step_size']
    data_params["stimuli_index"] = stimuli_index
    # load data
    loaders = load_data_AR(**data_params)
    train_loader_list, val_loader_list, test_loader_list, ar_train_loader, ar_val_loader, ar_test_loader, neurons_index = loaders
    
    deepAR_params  = cfg['deepAR']
    optim_params   = cfg['optimizer']
    deepAR_list = []
    deepAR_opt_list = []
    
    n_neurons = ar_train_loader.dataset.tensors[0].shape[-1]
    deepAR_param = deepAR_params['net']
    deepAR_param["num_rnn_inputs"] = n_neurons
    deepAR_lr    = deepAR_params['lr']
    deepAR_single = deepAR_net(**deepAR_param).to(device)
    opt_single = getattr(optim, optim_params['name'])(deepAR_single.parameters(), lr = deepAR_lr, eps=1e-8)

    # Multiple model for later model selection
    for train_loader in train_loader_list:
        n_neurons = train_loader.dataset.tensors[0].shape[-1]
        deepAR_param = deepAR_params['net']
        deepAR_param["num_rnn_inputs"] = n_neurons
        deepAR_lr    = deepAR_params['lr']
        deepAR = deepAR_net(**deepAR_param).to(device)
        deepAR_list.append(deepAR)
        optimizer_path = getattr(optim, optim_params['name'])(deepAR.parameters(), lr = deepAR_lr, eps=1e-8)
        deepAR_opt_list.append(optimizer_path)
    
    initialized = {
        "deepAR_single"    : deepAR_single,
        "opt_single"       : opt_single,
        "ar_train_loader"  : ar_train_loader, 
        "ar_val_loader"    : ar_val_loader, 
        'deepAR_list'      : deepAR_list,
        'train_loader_list': train_loader_list,
        'val_loader_list'  : val_loader_list,
        'deepAR_opt_list'  : deepAR_opt_list,
        'window_size'      : window_size,
        'step_size'        : step_size,
        'neurons_index'    : neurons_index,
        'n_epochs'         : cfg['n_epochs'],
        'device'           : device
    }
    return initialized, ar_test_loader, test_loader_list