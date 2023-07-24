import torch.nn as nn
import torch.optim as optim

from .att_encoder import *
from .flows import *
from .get_data import *
from torch import distributions as d
#from transformer import *


def setup_att_flow(cfg, important_index, device,run,seed,stimuli_index):
    data_params                    = cfg["data"]
    data_params["important_index"] = important_index
    data_params["seed"]            = seed
    data_params["test_run"]        = run
    data_params["stimuli_index"]   = stimuli_index
    time_resolution                = data_params["time_resolution"]
    filler                         = data_params["filler"]
    window_size                    = data_params["window_size"]
    target                         = data_params["target"]
    scaling_factor                 = data_params["scaling_factor"] if "scaling_factor" in data_params else 1
    sigma                          = data_params["sigma"] if "sigma" in data_params else 0.1
    optim_params                   = cfg["optimizer"]
    encoder_params                 = cfg["att_encoder"]
    flow_params                    = cfg["flow_net"]
    
    
    # load data
    train_loader, val_loader, test_loader, val_data_spike, val_data_smooth, val_q, data_spike, data_smooth, q = load_data_flow(**data_params)
    
    n_neurons = train_loader.dataset.tensors[1].shape[-1]
    stimuli_dim = train_loader.dataset.tensors[2].shape[-1]
    time_dim = train_loader.dataset.tensors[4].shape[-1]
    
    encoder_params['net']["num_rnn_inputs"] = n_neurons
    encoder_params['net']["attention"] = True \
        if "attention" in encoder_params['net'] and encoder_params['net']["attention"] \
        else False
    flow_params["net"]["num_inputs"]      = 1
    flow_params["net"]["num_cond_inputs"] =  encoder_params["net"]["context_dense_size"] + time_dim\
        if encoder_params["net"]["attention"] \
        else encoder_params["net"]["num_rnn_hidden"] + time_dim
    
    #flow_params["net"]["num_cond_inputs"] = encoder_params["net"]["num_rnn_hidden"] + stimuli_dim + time_dim

    encoder_lr    = encoder_params['lr']
    encoder = att_encoder(**encoder_params['net']).to(device)
    # Initialize Normalizing Flow
    
    flow_lr            = flow_params["lr"]
    num_inputs         = flow_params["net"]["num_inputs"]
    num_hidden         = flow_params["net"]["num_hidden"]
    s_act              = flow_params["net"]["s_act"]
    t_act              = flow_params["net"]["t_act"]
    # conditional variable include context vector dim + neuron_indicator + stimuli_indicator
    num_cond_inputs = flow_params["net"]["num_cond_inputs"]
    mask = torch.arange(0, num_inputs) % 2
    mask = mask.to(device)
    
    modules = []
    # realnvp
    for _ in range(flow_params["net"]["num_blocks"]):
        modules += [
            CouplingLayer(num_inputs, num_hidden, mask, num_cond_inputs,
                s_act=s_act, t_act=t_act),
            BatchNormFlow(num_inputs)
        ]
        mask = 1 - mask
    flow_net = FlowSequential(*modules).to(device)
    flow_net.base = d.normal.Normal(torch.tensor([0.0]).to(device),
                                    torch.tensor([1.0]).to(device))
    optimizer = getattr(optim, optim_params['name'])([{'params': encoder.parameters(), 'lr': encoder_lr},
                                                      {'params': flow_net.parameters(), 'lr': flow_lr}],
                                                     eps=1e-5)
    linear_transform = None
    initialized = {
        "n_epochs"        : cfg["n_epochs"],
        "train_loader"    : train_loader, 
        "val_loader"      : val_loader, 
        "encoder"         : encoder, 
        "flow_net"        : flow_net,
        "linear_transform": linear_transform,
        "optimizer"       : optimizer,
        "window_size"     : window_size,
        "n_neurons"       : n_neurons,
        "data_spike"      : val_data_spike,
        "data_smooth"     : val_data_smooth,
        "q"               : val_q,
        "time_resolution" : time_resolution,
        "filler"          : filler,
        "target_neuron"   : target,
        "important_index" : important_index,
        "scaling_factor"  : scaling_factor,
        "sigma"           : sigma
    }
    return initialized, test_loader, data_spike, data_smooth, q