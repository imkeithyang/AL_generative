import pandas as pd
import numpy as np
import json
import torch
import itertools


def read_moth(path, time_resolution=3, neuron_type=None, pred_label=None, addtype=False):
    """read moth data given path

    Args:
        path (_type_): path of moth data
        time_resolution (_type_): hyper parameter of time resolution

    Returns:
        _type_: return the moth data in dataframe, and the neurons list
    """
    splitted = path.split("/")[-1].split("_")
    moth = splitted[0] + ("_{}".format(splitted[1]) if splitted[1][0]!='c' else '')
    data_moth = pd.read_csv(path, index_col=None).drop(columns=["Unnamed: 0"])
    neurons = list(data_moth.keys())[2:]
    neurons_out = set([])
    for i in range(data_moth.shape[0]):
        for j in neurons:
            if (pred_label is not None and pred_label[moth][j]['true'] == neuron_type) \
                or neuron_type is None:
                try:
                    data_moth.at[i,j] = np.round(json.loads(data_moth.iloc[i][j]), time_resolution)
                except TypeError:
                    if np.isnan(data_moth.iloc[i][j]):
                        data_moth.at[i,j] = []
                neurons_out.add(j if not addtype else j+pred_label[moth][j]['true'])
    neurons_out = list(neurons_out)
    neurons_out.sort()
    return data_moth, neurons_out


def make_spiketrain(df, stimuli, run, neurons, time_resolution, min_spike = 0, pre_stim = False, tot_time=1.2):
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
    tot_timestep = int(time_scale*tot_time)
    data_concat = np.zeros((tot_timestep,len(neurons)))
    data_stimuli = df[df["label_stim"] == stimuli]
    for j, neuron in enumerate(neurons):
        data_concat[np.array((time_scale*data_stimuli.iloc[run][neuron])).astype(int),j] = 1
        
    return data_concat, neurons

def gaussian_smoothing_spike(data_concat, time_resolution, sigma):
    """Gaussian Smooth the spikes based on their timing"""
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


def split_window_per_neuron_flow(data_concat, data_concat_smooth, important_index,sigma,
                            window_size=3, target_neuron=0, time_resolution=3, filler=-1,
                            get_last_inf = False, shuffle = False, seed = 42):
    """split the spikes into windows and the corresponding label (interarrival spikes)"""
    time_scale = 10**time_resolution
    if shuffle == True:
        np.random.seed(seed)
        np.random.shuffle(data_concat[:,target_neuron])
    spike_time_unscaled = np.nonzero(data_concat[:,target_neuron])[0]

    # get inter-arrival time of spike
    
    # smoothing spikes with gaussian kernel
    target_time = np.diff(spike_time_unscaled)/time_scale
    source_time = []
    source_time_smooth = [] if data_concat_smooth is not None else None
    if important_index is None:
        important_index = list(range(data_concat.shape[1]))
    # Create window for spikes
    for t_index in range(len(target_time)):
        t = spike_time_unscaled[t_index]
        source_time_filter = data_concat[t-window_size:t,:]
        if data_concat_smooth is not None:
            source_time_filter_smooth = gaussian_smoothing_spike(data_concat[t-window_size:t,:], time_resolution, sigma)
            
        if t < window_size:
            source_time_filter = np.zeros((window_size,data_concat.shape[1]))
            source_time_filter[0:window_size - t] = filler
            source_time_filter[window_size - t:] = data_concat[0:t,:]
            if data_concat_smooth is not None:
                source_time_filter_smooth = np.zeros((window_size,data_concat_smooth.shape[1]))
                source_time_filter_smooth[0:window_size - t] = filler
                source_time_filter_smooth[window_size - t:] = gaussian_smoothing_spike(data_concat[0:t,:], time_resolution, sigma)
            
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
    if len(spike_time_unscaled) > 1 and get_last_inf == True:
        temp = np.zeros((len(target_time)+1,))
        temp[-1] = 1
        temp[0:-1] = target_time
        target_time = temp
        
        # If things are done correctly t_index+1 should be the last spike
        try:
            t = spike_time_unscaled[t_index+1]
        except:
            t = spike_time_unscaled[0]
        source_time_filter = data_concat[t-window_size:t,:]
        source_time.append(np.array(source_time_filter[:,important_index]))
        if data_concat_smooth is not None:
            source_time_filter_smooth = gaussian_smoothing_spike(data_concat[t-window_size:t,:], time_resolution, sigma)
            source_time_smooth.append(np.array(source_time_filter_smooth[:,important_index]))
    

    time_conditional = spike_time_unscaled/time_scale
    if time_conditional.size:
        time_conditional[1:] = time_conditional[0:-1]
        time_conditional[0] = 0
        if get_last_inf:
            time_cond_temp = np.zeros((len(time_conditional)+1,))
            time_cond_temp[0:-1] = time_conditional
            time_cond_temp[-1] = (len(data_concat)-1)/time_scale
            time_conditional = time_cond_temp
    else:
        time_conditional = np.array([0])
    return np.array(target_time)[..., np.newaxis], np.array(source_time), np.array(source_time_smooth), np.array(time_conditional)[..., np.newaxis]

def split_all_stimuli_flow(df, neurons, target,
                           time_resolution,stimuli_index, important_index, test_run, window_size=3, 
                           min_spike = 0, sigma=0.001, filler=-1,pre_stim=False, use_component=False, shuffle = False, seed = 42):
    
    # split all stimuli's data, there are multiple runs involved as well
    components =["0-BEA","0-BOL","0-MAL","0-MYR","0-LIN","0-NER","0-GER","0-ISO","0-FAR", "0-DATEXT", "0-CTL"]
    mixture_components = {"1-P9":["0-BEA","0-BOL","0-MAL","0-MYR","0-LIN","0-NER","0-GER","0-ISO","0-FAR"],
                  "1-P9_TEN":["0-BEA","0-BOL","0-MAL","0-MYR","0-LIN","0-NER","0-GER","0-ISO","0-FAR"],
                  "1-P9_HUND":["0-BEA","0-BOL","0-MAL","0-MYR","0-LIN","0-NER","0-GER","0-ISO","0-FAR"],
                  "1-P9_TENTHOUS":["0-BEA","0-BOL","0-MAL","0-MYR","0-LIN","0-NER","0-GER","0-ISO","0-FAR"],
                  "1-P5":["0-BEA","0-BOL","0-LIN","0-NER","0-GER"],
                  "1-P4":["0-BEA","0-BOL","0-LIN","0-NER"],
                  "1-P3":["0-BEA","0-BOL","0-LIN"],
                  "0-M6":["0-MAL","0-MYR","0-NER","0-GER","0-ISO","0-FAR"],
                  "0-M5":["0-MAL","0-MYR","0-GER","0-ISO","0-FAR"],
                  "0-M4":["0-MAL","0-MYR","0-ISO","0-FAR"],
                  "0-M3":["0-MAL","0-ISO","0-FAR"],
                  "0-M2":["0-BEA","0-BOL"]}
    
    df[["label", "stimuli"]].drop_duplicates()
    df["label_stim"] = df["label"].astype(str) + "-" + df["stimuli"].str.upper()
    all_stimuli_count = df.value_counts("label_stim").to_dict()
    all_stimuli_count = dict(sorted(all_stimuli_count.items()))
    keys = list(all_stimuli_count.keys())
    #for key in keys:
        #if "DAT+" in key or "BENZALDEHYE" in key:
        #    all_stimuli_count.pop(key)
    num_stimuli = len(all_stimuli_count) - 12 if use_component else len(all_stimuli_count)
    val_run = test_run - 1
    
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
    stim_name = []
    for s_index, s in enumerate(all_stimuli_count):
        if s_index not in set(list(stimuli_index) if isinstance(stimuli_index, range) else [stimuli_index]):
            continue
        # One hot encoding with components
        if use_component and s in mixture_components:
            comp = mixture_components[s]
            s_index = []
            for c in comp:
                s_index.append(np.where(np.array(components) == c)[0])
        elif use_component and s in set(components):
            s_index = np.where(np.array(components) == s)[0]
        # special case to deal with incomplete dataset
        
        stim_name.append(s)
        for run in range(all_stimuli_count[s]):
            if val_run < 0 or val_run > all_stimuli_count[s]-1:
                val_run = all_stimuli_count[s]-1
            if test_run > all_stimuli_count[s]-1:
                test_run = 0
            data_concat_has_spike, neurons_has_spike = make_spiketrain(df, 
                                                                       s,
                                                                       run,
                                                                       neurons, 
                                                                       time_resolution, 
                                                                       min_spike,
                                                                       pre_stim)
            
            # Smoothing of the spikes
            data_concat_smooth = gaussian_smoothing_spike(data_concat_has_spike,time_resolution,sigma)
            
            
            # split for target autoregressive
            extracted_data = split_window_per_neuron_flow(data_concat_has_spike, 
                                                          data_concat_smooth,
                                                          important_index,sigma,
                                                          window_size, target, time_resolution, filler,shuffle = shuffle, seed = seed)
            ar_target_interarrival, ar_windowed_spike, ar_windowed_smooth, time_conditional  = extracted_data
            if run == val_run:
                print('')
                pass
            # splitting the cases for training validating and testing dataset
            # For 5 runs, 3 run will be used as training, 1 run for validating, 1 run for testing
            # This is done in a rotational bases so every run could be used as a testing run
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
                
    # Concatenating all data
    training_data = [ar_window_spike, ar_window_smooth, ar_stimuli, ar_target, ar_time]
    validating_data = [ar_val_window_spike, ar_val_window_smooth, ar_val_stimuli, ar_val_target, ar_val_time]
    testing_data = [ar_test_window_spike, ar_test_window_smooth, ar_test_stimuli, ar_test_target, ar_test_time]
    
    val_generative_comparison = [val_data_concat_window_spike, 
                                 val_data_concat_window_smooth,
                                 val_data_concat_stimuli]
    
    generative_comparison = [data_concat_window_spike, data_concat_window_smooth, data_concat_stimuli]
            
    return  training_data, validating_data, testing_data, val_generative_comparison, generative_comparison, stim_name


def load_data_flow(path, 
                   target,
                   time_resolution, 
                   window_size, 
                   important_index,
                   test_run,
                   stimuli_index = None,
                   minimum_spike_count=0, 
                   batch_size=128, seed=42, sigma=0.1, 
                   scaling_factor=1, filler=-1,
                   use_component=False,
                   neuron_type=None,
                   shuffle = False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    pre_stim = ("pre_stimuli" in path)
    try:
        pred = json.load(open("unlabeled_pred.json",'r'))
        if "011124" in path or "12142022" in path:
            pred = None
    except:
        pred = None
    data_moth, neurons = read_moth(path, time_resolution, neuron_type=neuron_type, pred_label=pred)
    try:
        data_moth = data_moth.drop("Channel13d", axis=1)
        neurons.pop()
    except:
        pass
    if stimuli_index is None:
        stimuli_index = range(0,23)
    if target > len(neurons):
        raise IndexError("Target index is greater than the number of neurons")
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
                                       pre_stim,
                                       use_component=use_component, shuffle = shuffle, seed = seed)
    training_data, validating_data, testing_data, val_generative_comparison, generative_comparison, stim_name = extracted
    
    ar_window_spike = training_data[0]
    ar_window_smooth = training_data[1]
    ar_stimuli = training_data[2]
    ar_target = training_data[3]
    ar_time = training_data[4]
    
    # Make loaders
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(ar_window_spike).float(),
                                               torch.from_numpy(ar_window_smooth).float(),
                                               torch.from_numpy(ar_stimuli).float(),
                                               torch.from_numpy(ar_target).float()*scaling_factor,
                                               torch.from_numpy(ar_time).float()*scaling_factor)
    
    ar_train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    
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
    return ar_train_loader, ar_val_loader, ar_test_loader, val_data_concat_has_spike, val_data_concat_smooth, val_data_concat_stimuli, data_concat_has_spike, data_concat_smooth, data_concat_stimuli, stim_name, neurons