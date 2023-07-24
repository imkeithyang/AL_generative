import pandas as pd
import numpy as np
import json
import torch

from .get_data_AR import read_moth, make_spiketrain, gaussian_smoothing_spike

def split_window_per_neuron_flow(data_concat, data_concat_smooth, important_index,
                            window_size=3, target_neuron=0, time_resolution=3, filler=-1,
                            get_last_inf = False):
    time_scale = 10**time_resolution
    spike_time_unscaled = np.nonzero(data_concat[:,target_neuron])[0]
    # get inter-arrival time of spike
    
    # smoothing spikes with gaussian kernel
    target_time = np.diff(spike_time_unscaled)/time_scale
    source_time = []
    source_time_smooth = [] if data_concat_smooth is not None else None
    
    # Create window for spikes
    for t_index in range(len(target_time)):
        t = spike_time_unscaled[t_index]
        source_time_filter = data_concat[t-window_size:t,:]
        if data_concat_smooth is not None:
            source_time_filter_smooth = data_concat_smooth[t-window_size:t,:]
            
        if t < window_size:
            source_time_filter = np.zeros((window_size,data_concat.shape[1]))
            source_time_filter[0:window_size - t] = filler
            source_time_filter[window_size - t:] = data_concat[0:t,:]
            if data_concat_smooth is not None:
                source_time_filter_smooth = np.zeros((window_size,data_concat_smooth.shape[1]))
                source_time_filter_smooth[0:window_size - t] = filler
                source_time_filter_smooth[window_size - t:] = data_concat_smooth[0:t,:]
            
        source_time.append(np.array(source_time_filter[:,important_index]))
        if data_concat_smooth is not None:
            source_time_smooth.append(np.array(source_time_filter_smooth[:,important_index]))
        
    # First target time should be spike_time_unscaled[0] with source_time_filter = filler
    # If there is no spike, then target time should be large
    temp = np.zeros((len(target_time)+1,))
    temp[0] = spike_time_unscaled[0]/time_scale if spike_time_unscaled.size else 2
    temp[1:] = target_time
    target_time = temp
    source_time.insert(0, np.zeros((window_size,len(important_index)))+filler)
    if data_concat_smooth is not None:
        source_time_smooth.insert(0, np.zeros((window_size,len(important_index)))+filler)
    
    
    # There should be a spike with interarrival time = inf that indicates the end of process at the very end
    if len(spike_time_unscaled) > 0 and get_last_inf == True:
        temp = np.zeros((len(target_time)+1,))
        temp[-1] = 1
        temp[0:-1] = target_time
        target_time = temp
        
        # If things are done correctly t_index+1 should be the last spike
        t = spike_time_unscaled[t_index+1]
        source_time_filter = data_concat[t-window_size:t,:]
        if data_concat_smooth is not None:
            source_time_filter_smooth = data_concat_smooth[t-window_size:t,:]
        
        source_time.append(np.array(source_time_filter[:,important_index]))
        if data_concat_smooth is not None:
            source_time_smooth.append(np.array(source_time_filter_smooth[:,important_index]))
    

    time_conditional = spike_time_unscaled/time_scale
    if time_conditional.size:
        time_conditional[1:] = time_conditional[0:-1]
        time_conditional[0] = 0
    else:
        time_conditional = np.array([0])
    return np.array(target_time)[..., np.newaxis], np.array(source_time), np.array(source_time_smooth), np.array(time_conditional)[..., np.newaxis]

def split_all_stimuli_flow(df, neurons, target,
                           time_resolution, stimuli_index,
                           important_index, test_run, window_size=3, 
                           min_spike = 0, sigma=0.001, filler=-1,pre_stim=False):
    all_stimuli_count = df.value_counts("stimuli").to_dict()
    num_stimuli = len(all_stimuli_count)
    val_run = test_run - 1
    neurons_index = np.zeros(shape=(len(important_index)))
    ar_stimuli = np.array([])
    ar_window_spike = np.array([])
    ar_window_smooth = np.array([])
    ar_target = np.array([])
    ar_time = np.array([])
    
    ar_val_window_spike = np.array([])
    ar_val_window_smooth = np.array([])
    ar_val_stimuli = np.array([])
    ar_val_target = np.array([])
    ar_val_time = np.array([])
    
    ar_test_stimuli = np.array([])
    ar_test_window_spike = np.array([])
    ar_test_window_smooth = np.array([])
    ar_test_target = np.array([])
    ar_test_time = np.array([])
    
    val_data_concat_stimuli = []
    val_data_concat_window_spike = []
    val_data_concat_window_smooth = []
    
    data_concat_stimuli = []
    data_concat_window_spike = []
    data_concat_window_smooth = []
    data_concat_target = []
    

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
                                                                       min_spike,
                                                                       pre_stim)
            
            data_concat_has_spike
            # Smoothing of the spikes
            data_concat_smooth = gaussian_smoothing_spike(data_concat_has_spike,time_resolution,sigma)
            
            
            # split for target autoregressive
            extracted_data = split_window_per_neuron_flow(data_concat_has_spike, 
                                                          data_concat_smooth,
                                                          important_index,
                                                          window_size, target, time_resolution, filler)
            ar_target_interarrival, ar_windowed_spike, ar_windowed_smooth, time_conditional  = extracted_data
            
            if ar_target_interarrival.size and run != val_run and run != test_run: 
                ar_window_spike = np.vstack([ar_window_spike, 
                                    ar_windowed_spike]) if ar_window_spike.size else ar_windowed_spike
                ar_window_smooth = np.vstack([ar_window_smooth, 
                                    ar_windowed_smooth]) if ar_window_smooth.size else ar_windowed_smooth
                ar_target = np.vstack([ar_target, 
                                    ar_target_interarrival]) if ar_target.size else ar_target_interarrival
                ar_time = np.vstack([ar_time, 
                                    time_conditional]) if ar_time.size else time_conditional
                
                stimuli_rep = np.zeros(shape=(ar_target_interarrival.shape[0], num_stimuli))
                stimuli_rep[:,s_index] = 1
                ar_stimuli = np.vstack([ar_stimuli, 
                                    stimuli_rep]) if ar_stimuli.size else stimuli_rep
                
            elif ar_target_interarrival.size and run == val_run: 
                ar_val_window_spike = np.vstack([ar_val_window_spike, 
                                    ar_windowed_smooth]) if ar_val_window_spike.size else ar_windowed_smooth
                ar_val_window_smooth = np.vstack([ar_val_window_smooth, 
                                    ar_windowed_smooth]) if ar_val_window_smooth.size else ar_windowed_smooth
                ar_val_target = np.vstack([ar_val_target, 
                                    ar_target_interarrival]) if ar_val_target.size else ar_target_interarrival
                ar_val_time = np.vstack([ar_val_time, 
                                    time_conditional]) if ar_val_time.size else time_conditional
                
                stimuli_rep = np.zeros(shape=(ar_target_interarrival.shape[0], num_stimuli))
                stimuli_rep[:,s_index] = 1
                ar_val_stimuli = np.vstack([ar_val_stimuli, 
                                    stimuli_rep]) if ar_val_stimuli.size else stimuli_rep
                stimu = np.zeros(shape=(1, num_stimuli))
                stimu[:,s_index] = 1
                val_data_concat_stimuli.append(stimu)
                val_data_concat_window_spike.append(data_concat_has_spike)
                val_data_concat_window_smooth.append(data_concat_smooth)
                
            elif ar_target_interarrival.size and run == test_run: 
                ar_test_window_spike = np.vstack([ar_test_window_spike, 
                                    ar_windowed_smooth]) if ar_test_window_spike.size else ar_windowed_smooth
                ar_test_window_smooth = np.vstack([ar_test_window_smooth, 
                                    ar_windowed_smooth]) if ar_test_window_smooth.size else ar_windowed_smooth
                ar_test_target = np.vstack([ar_test_target, 
                                    ar_target_interarrival]) if ar_test_target.size else ar_target_interarrival
                ar_test_time = np.vstack([ar_test_time, 
                                    time_conditional]) if ar_test_time.size else time_conditional
                stimuli_rep = np.zeros(shape=(ar_target_interarrival.shape[0], num_stimuli))
                stimuli_rep[:,s_index] = 1
                ar_test_stimuli = np.vstack([ar_test_stimuli, 
                                    stimuli_rep]) if ar_test_stimuli.size else stimuli_rep
                stimu = np.zeros(shape=(1, num_stimuli))
                stimu[:,s_index] = 1
                data_concat_stimuli.append(stimu)
                data_concat_window_spike.append(data_concat_has_spike)
                data_concat_window_smooth.append(data_concat_smooth)
                
    training_data = [ar_window_spike, ar_window_smooth, ar_stimuli, ar_target, ar_time]
    validating_data = [ar_val_window_spike, ar_val_window_smooth, ar_val_stimuli, ar_val_target, ar_val_time]
    testing_data = [ar_test_window_spike, ar_test_window_smooth, ar_test_stimuli, ar_test_target, ar_test_time]
    
    val_generative_comparison = [val_data_concat_window_spike, 
                                 val_data_concat_window_smooth,
                                 val_data_concat_stimuli]
    
    generative_comparison = [data_concat_window_spike, data_concat_window_smooth, data_concat_stimuli]
            
    return  training_data, validating_data, testing_data, generative_comparison, val_generative_comparison


def load_data_flow(path, 
                   target,
                   time_resolution, 
                   window_size,
                   stimuli_index,
                   important_index,
                   test_run,
                   minimum_spike_count=0, 
                   batch_size=128, seed=0, sigma=0.1, scaling_factor=1, filler=-1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pre_stim = ("pre_stimuli" in path)
    data_moth, neurons = read_moth(path, time_resolution)
    extracted = split_all_stimuli_flow(data_moth, 
                                       neurons, 
                                       target,
                                       time_resolution,
                                       stimuli_index,
                                       important_index,
                                       test_run,
                                       window_size, 
                                       minimum_spike_count,
                                       sigma,
                                       filler,
                                       pre_stim)
    training_data, validating_data, testing_data, val_generative_comparison, generative_comparison = extracted
    
    ar_window_spike = training_data[0]
    ar_window_smooth = training_data[1]
    ar_stimuli = training_data[2]
    ar_target = training_data[3]
    ar_time = training_data[4]
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(ar_window_spike).float(),
                                               torch.from_numpy(ar_window_smooth).float(),
                                               torch.from_numpy(ar_stimuli).float(),
                                               torch.from_numpy(ar_target).float()*scaling_factor,
                                               torch.from_numpy(ar_time).float()*scaling_factor)
    
    drop_last = False if ar_window_spike.shape[0] < batch_size else True
    ar_train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    
    ar_val_window_spike = validating_data[0]
    ar_val_window_smooth = validating_data[1]
    ar_val_stimuli = validating_data[2]
    ar_val_target = validating_data[3]
    ar_val_time = validating_data[4]
    val_data = torch.utils.data.TensorDataset(torch.from_numpy(ar_val_window_spike).float(),
                                               torch.from_numpy(ar_val_window_smooth).float(),
                                               torch.from_numpy(ar_val_stimuli).float(),
                                               torch.from_numpy(ar_val_target).float()*scaling_factor,
                                               torch.from_numpy(ar_val_time).float()*scaling_factor)
    
    ar_val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
    
    ar_test_window_spike = testing_data[0]
    ar_test_window_smooth = testing_data[1]
    ar_test_stimuli = testing_data[2]
    ar_test_target = testing_data[3]
    ar_test_time = testing_data[4]
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(ar_test_window_spike).float(),
                                               torch.from_numpy(ar_test_window_smooth).float(),
                                               torch.from_numpy(ar_test_stimuli).float(),
                                               torch.from_numpy(ar_test_target).float()*scaling_factor,
                                               torch.from_numpy(ar_test_time).float()*scaling_factor)
    ar_test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
    
    val_data_concat_has_spike = val_generative_comparison[0] 
    val_data_concat_smooth    = val_generative_comparison[1]
    val_data_concat_stimuli   = val_generative_comparison[2]
    
    data_concat_has_spike = generative_comparison[0] 
    data_concat_smooth = generative_comparison[1]
    data_concat_stimuli = generative_comparison[2]
    
    return ar_train_loader, ar_val_loader, ar_test_loader, val_data_concat_has_spike, val_data_concat_smooth, val_data_concat_stimuli, data_concat_has_spike, data_concat_smooth, data_concat_stimuli