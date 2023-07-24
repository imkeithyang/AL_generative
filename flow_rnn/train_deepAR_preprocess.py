import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from .deepAR import *
from .setup_deepAR import *

def train_single_deepAR(deepAR,
                        path_loader,
                        validation_loader,
                        window_size,
                        step_size,
                        optimizer_path,
                        n_epochs,
                        device):
    
    deepAR_loss_list    = []
    val_loss_list       = []
    
    pbar = tqdm(total=n_epochs)

    
    best_epoch  = 0
    val_loss = 0
    for i in range(n_epochs):
        train_loss = 0
        deepAR.train()
        g_loss = nn.MSELoss()
        for batch_idx, data in enumerate(path_loader):
            hidden=None
            optimizer_path.zero_grad() 
            
            spike = data[0].float().to(device)
            smooth = data[1].float().to(device)
            q = data[2].float().to(device)
            target = data[3].float().to(device)
            
            optimizer_path.zero_grad() 
            spike_count_hat, hidden = deepAR(spike,q=None, t=None, hidden=None)
            loss = g_loss(spike_count_hat.flatten(), target.flatten())
            
            loss.backward()
            optimizer_path.step()
            train_loss += loss.item()
            pbar.set_description("E-B: {}-{} | deepAR Loss: {} | Val Loss: {} | Avg Target {} | Best Epoch {}".format(
                i+1, 
                batch_idx+1,
                round(train_loss/(batch_idx + 1), 2),
                round(val_loss,2),
                round(target.detach().mean().item()),
                best_epoch
            ))
            
        deepAR_loss = train_loss/(batch_idx + 1)
        deepAR_loss_list.append(deepAR_loss)
        # Validate
        if (i+1)%1 == 0:
            deepAR.eval()
            
            val_loss = validate_deepAR(deepAR,
                                       device,
                                       validation_loader)
            if len(val_loss_list) > 0:
                if np.min(val_loss_list) > val_loss:
                    best_epoch = i + 1
            val_loss_list.append(val_loss)
        pbar.update(1)
        
        # Early Stopping
        if i - best_epoch > 20:
            break
        
    pbar.close()
    stats = {"deepAR_loss":deepAR_loss_list,
             "val_loss": val_loss_list, 'best_epoch':best_epoch}
    return stats

def validate_deepAR(deepAR,
                    device,
                    validation_loader,
                    **params):
    deepAR.eval()
    with torch.no_grad():
        g_loss = nn.MSELoss()
        val_loss = 0
        temp = []
        for batch_idx, data in enumerate(validation_loader):
            hidden=None
            
            spike = data[0].float().to(device)
            smooth = data[1].float().to(device)
            q = data[2].float().to(device)
            target = data[3].float().to(device)
            spike_count_hat, hidden = deepAR(spike, q=None, t=None, hidden=None)
            loss = g_loss(spike_count_hat.flatten(), target.flatten())
            val_loss += loss.item()

        val_loss /= (batch_idx + 1)
    return val_loss

def train_multiple_deepAR(deepAR_single,
                          opt_single,
                          ar_train_loader,
                          ar_val_loader,
                          deepAR_list,
                          train_loader_list,
                          val_loader_list,
                          deepAR_opt_list,
                          window_size,
                          step_size,
                          neurons_index,
                          n_epochs,     
                          device):

    target = neurons_index[0,0]
    pairs = neurons_index[:,1]
    store_loss = np.zeros((neurons_index.shape[0] + 1,))
    stats_dict = {}
    
    ar_single_stats = train_single_deepAR(deepAR_single,
                                    ar_train_loader,
                                    ar_val_loader,
                                    window_size,
                                    step_size,
                                    opt_single,
                                    n_epochs, 
                                    device)
    stats_dict[target] = ar_single_stats
    ar_single_best_loss = np.min(ar_single_stats["val_loss"])
    store_loss[target] += ar_single_best_loss
        
    for deepAR, train_loader, val_loader, opt, pair_index in zip(deepAR_list, train_loader_list, 
                                                                val_loader_list, deepAR_opt_list, pairs):
        print("Training on Neuron Pair {}".format(pair_index))
        ar_stats = train_single_deepAR(deepAR,
                                    train_loader,
                                    val_loader,
                                    window_size,
                                    step_size,
                                    opt,
                                    n_epochs, 
                                    device)
        stats_dict[pair_index] = ar_stats
        ar_best_loss = np.min(ar_stats["val_loss"])
        store_loss[pair_index] += ar_best_loss
    return stats_dict, store_loss



    
    
    