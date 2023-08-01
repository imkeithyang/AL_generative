import pandas as pd
import numpy as np
import json
import torch
import itertools


def read_moth(path, time_resolution=3):
    """read moth data given path

    Args:
        path (_type_): path of moth data
        time_resolution (_type_): hyper parameter of time resolution

    Returns:
        _type_: return the moth data in dataframe, and the neurons list
    """
    data_moth = pd.read_csv(path, index_col=0)
    neurons = list(data_moth.keys())[2:]
    for i in range(data_moth.shape[0]):
        for j in neurons:
            data_moth.at[i,j] = np.round(json.loads(data_moth.iloc[i][j]), time_resolution)
    return data_moth, neurons


def make_spiketrain(df, stimuli, run, neurons, time_resolution, min_spike = 0, pre_stim = False):
    """_summary_

    Args:
        df (_type_): moth data
        stimuli (_type_): particular stimuli
        run (_type_): each stimuli has 5 runs
        neurons (_type_): neurons list
        time_resolution (_type_): time resolution
        min_spike (int, optional): filter the minmum amount of spikes. Defaults to 0.

    Returns:
        _type_: filtered spike train (has spike = 1, no spike = 0), filtered neurons
    """
    
    time_scale = 10**time_resolution
    tot_timestep = int(time_scale*1.2)
    data_concat = np.zeros((tot_timestep,len(neurons)))
    data_stimuli = df[df["stimuli"] == stimuli]
    for j, neuron in enumerate(neurons):
        data_concat[(time_scale*data_stimuli.iloc[run][neuron]).astype(int),j] = 1
        
    return data_concat, neurons
    
def split_window_per_neuron(data_concat, data_concat_smooth, 
                            window_size=3, target_neuron=0, time_resolution=3, filler=-1):
    time_scale = 10**time_resolution
    windowed_spike = data_concat.reshape(-1,window_size,data_concat.shape[-1])
    windowed_smooth = data_concat_smooth.reshape(-1,window_size,data_concat.shape[-1])
    
    # Create window for spikes
    target = windowed_spike.sum(1)[..., 0][1:]
    windowed_spike = windowed_spike[0:-1]
    windowed_smooth = windowed_smooth[0:-1]
    
    return np.array(target)[..., np.newaxis], windowed_spike, windowed_smooth

def split_window_all_neurons(data_concat, data_concat_smooth, neurons, window_size, target,
                             time_resolution, filler):

    windowed_spike = []
    windowed_smooth = []
    target_interarrival = []
    
    for i in range(len(neurons)):
        n_index = neurons[i].astype(int)
        target_spike_count, source_time, source_time_smooth = split_window_per_neuron(data_concat[:,n_index],
                                                                               data_concat_smooth[:,n_index],
                                                                               window_size, 
                                                                               target, 
                                                                               time_resolution, 
                                                                               filler)
        windowed_spike.append(source_time)
        windowed_smooth.append(source_time_smooth)
        target_interarrival.append(target_spike_count)
        
    return np.array(windowed_spike), np.array(windowed_smooth), np.array(target_interarrival)

def gaussian_smoothing_spike(data_concat, time_resolution, sigma):
    time_scale = 10**time_resolution
    tot_time = data_concat.shape[0]/time_scale
    data_concat_smooth = []
    for n in range(data_concat.shape[-1]):
        cat_temp = np.zeros(data_concat[:,n].shape).reshape(-1,1)
        spike_time_unscaled = np.nonzero(data_concat[:,n])[0]
        target_time = np.diff(spike_time_unscaled)/time_scale
        if spike_time_unscaled.size != 0:
            for t in spike_time_unscaled:
                cat_temp += np.exp(-((np.arange(0,tot_time,1/time_scale) - t/time_scale)/sigma)**2).reshape(-1,1)
        data_concat_smooth.append(cat_temp)
    data_concat_smooth = np.concatenate(data_concat_smooth, 1)
    return data_concat_smooth
        
def split_all_stimuli(df, neurons, target,stimuli_index,
                      time_resolution, test_run, window_size=3, 
                      min_spike = 0, sigma=0.001, filler=-1):
    
    all_stimuli_count = df.value_counts("stimuli").to_dict()
    val_run = test_run - 1
    spike_list, smooth_list, stimuli_list = [],[],[]
    val_spike_list, val_smooth_list, val_stimuli_list = [],[],[]
    neuron_index = list(range(len(neurons)))
    neuron_pairs = np.zeros((len(neuron_index)-1, 2))
    neuron_pairs[:,0] = target
    neuron_pairs[:,1] = neuron_index[:target] + neuron_index[target+1:]
    neuron_pairs = neuron_pairs.astype(int)
    for i, n in enumerate(neurons):
        if i == 0:
            index = target
        else:
            index = neuron_pairs[i-1]

        data_spike       = []
        data_smooth      = []
        data_stimuli     = []
        val_data_spike   = []
        val_data_smooth  = []
        val_data_stimuli = []
        for s_index, s in enumerate(all_stimuli_count):
            if s_index != stimuli_index:
                continue
            for run in range(all_stimuli_count[s]):
                if val_run < 0:
                    val_run = all_stimuli_count[s]-1
                data_concat_has_spike, neurons_has_spike = make_spiketrain(df, 
                                                                           s,
                                                                           run,
                                                                           neurons, 
                                                                           time_resolution, 
                                                                           min_spike)
                
                data_concat_smooth = gaussian_smoothing_spike(data_concat_has_spike, time_resolution, sigma)
                
                data_concat_has_spike = data_concat_has_spike[..., index]
                data_concat_smooth = data_concat_smooth[..., index]
                
                if i == 0:
                    data_concat_has_spike = data_concat_has_spike.reshape(-1,1)
                    data_concat_smooth = data_concat_smooth.reshape(-1,1)
                    
                if run != val_run:
                    data_stimuli.append(np.array([s_index]))
                    data_spike.append(data_concat_has_spike) 
                    data_smooth.append(data_concat_smooth)
                else:                 
                    val_data_stimuli.append(np.array([s_index]))
                    val_data_spike.append(data_concat_has_spike) 
                    val_data_smooth.append(data_concat_smooth)
                        
        spike_list.append(np.array(data_spike))
        smooth_list.append(np.array(data_smooth))
        stimuli_list.append(np.array(data_stimuli))
        
        val_spike_list.append(np.array(val_data_spike))
        val_smooth_list.append(np.array(val_data_smooth))
        val_stimuli_list.append(np.array(val_data_stimuli))
    
    training_data = [spike_list, 
                     smooth_list, 
                     stimuli_list]
    
    validating_data = [val_spike_list, 
                       val_smooth_list,
                       val_stimuli_list]

    return  training_data, validating_data, neuron_pairs

def load_data_AR(path, time_resolution, window_size, step_size, test_run,target,stimuli_index, minimum_spike_count=0, 
              batch_size=128, seed=0, sigma=0.1, filler=-1):
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_moth, neurons = read_moth(path, time_resolution)
    
    
    extracted = split_all_stimuli(data_moth, 
                                  neurons, 
                                  target,
                                  stimuli_index,
                                  time_resolution,
                                  test_run, 
                                  window_size,
                                  minimum_spike_count,
                                  sigma,
                                  filler)
    training_data, validating_data, neurons_index = extracted
    
    window_spike_list, window_smooth_list, stimuli_list = \
        training_data[0], training_data[1], training_data[2]
    
    val_window_spike_list, val_window_smooth_list, val_stimuli_list = \
        validating_data[0], validating_data[1], validating_data[2]
    
    train_loader_list = []
    val_loader_list = []
    for i in range(len(window_spike_list)):
        window_spike_unfold = torch.from_numpy(window_spike_list[i]).float().unfold(1,window_size, step_size)
        window_smooth_unfold = torch.from_numpy(window_smooth_list[i]).float().unfold(1,window_size, step_size)
        target_list = window_spike_unfold[:,1:, 0,:].sum(-1)
        window_spike_unfold = torch.transpose(window_spike_unfold, -2, -1)[:,0:-1]
        window_smooth_unfold = torch.transpose(window_smooth_unfold, -2, -1)[:,0:-1]
        stimuli_list_unfold = torch.from_numpy(stimuli_list[i]).float().repeat((1,window_spike_unfold.shape[1]))
        stimuli_list_unfold = stimuli_list_unfold.unsqueeze(-1)
        train_data = torch.utils.data.TensorDataset(torch.flatten(window_spike_unfold,0,1),
                                                    torch.flatten(window_smooth_unfold,0,1),
                                                    torch.flatten(stimuli_list_unfold,0,1),
                                                    torch.flatten(target_list,0,1))
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        
        val_window_spike_unfold = torch.from_numpy(val_window_spike_list[i]).float().unfold(1,window_size, step_size)
        val_window_smooth_unfold = torch.from_numpy(val_window_smooth_list[i]).float().unfold(1,window_size, step_size)
        val_target_list = val_window_spike_unfold[:,1:, 0,:].sum(-1)
        val_window_spike_unfold = torch.transpose(val_window_spike_unfold, -2, -1)[:,0:-1]
        val_window_smooth_unfold = torch.transpose(val_window_smooth_unfold, -2, -1)[:,0:-1]
        val_stimuli_list_unfold = torch.from_numpy(val_stimuli_list[i]).float().repeat((1,window_spike_unfold.shape[1]))
        val_stimuli_list_unfold = val_stimuli_list_unfold.unsqueeze(-1)
        val_data = torch.utils.data.TensorDataset(torch.flatten(val_window_spike_unfold,0,1),
                                                  torch.flatten(val_window_smooth_unfold,0,1),
                                                  torch.flatten(val_stimuli_list_unfold,0,1),
                                                  torch.flatten(val_target_list,0,1))
        
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
        
        train_loader_list.append(train_loader)
        val_loader_list.append(val_loader)
    
    ar_train_loader = train_loader_list[0]
    ar_val_loader = val_loader_list[0]
    
    train_loader_list = train_loader_list[1:]
    val_loader_list = val_loader_list[1:]
    
    ar_test_loader = []
    test_loader_list = []
    return train_loader_list, val_loader_list, test_loader_list, ar_train_loader, ar_val_loader, ar_test_loader, neurons_index
