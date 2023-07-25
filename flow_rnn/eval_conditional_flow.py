import torch
import pyspike as spk
import numpy as np
from .get_data import split_window_per_neuron_flow
from .get_data_AR import gaussian_smoothing_spike

def evaluate_spike_distance(data_spike, gen_spike_list, target_neuron, time_resolution):
    time_scale = 10**time_resolution
    target_spike_train_emp = np.nonzero(data_spike[:,target_neuron])[0]/time_scale
    edges=(0,data_spike.shape[0]/time_scale)
    
    spike_train_emp = spk.SpikeTrain(target_spike_train_emp,edges)
    spike_train_gen_list = []
    for gen_spike in gen_spike_list:
        target_spike_train_gen = np.nonzero(gen_spike[:,target_neuron])[0]/time_scale
        spike_train_gen_list.append(spk.SpikeTrain(target_spike_train_gen, edges))
        
    isi_dist_list   = []
    spike_dist_list = []
    for spike_train_gen in spike_train_gen_list:
        isi_dist = spk.isi_distance(spike_train_emp, spike_train_gen)
        spike_dist = spk.spike_distance(spike_train_emp, spike_train_gen)
        isi_dist_list.append(isi_dist)
        spike_dist_list.append(spike_dist)
        
    return isi_dist_list, spike_dist_list

def evaluate_spikesync_unit_pairs(data_spike, target_neuron, time_resolution):
    time_scale = 10**time_resolution
    tar_spike_train = np.nonzero(data_spike[:,target_neuron])[0]/time_scale
    
    edges=(0,data_spike.shape[0]/time_scale)
    tar_spike_train = spk.SpikeTrain(tar_spike_train, edges)
    spike_sync_mat = np.zeros_like(data_spike).T
    for i in range(data_spike.shape[1]):
        if i == target_neuron:
            continue
        ref_spike_train = spk.SpikeTrain(np.nonzero(data_spike[:,i])[0]/time_scale, edges)
        spike_sync = spk.spike_sync_profile(tar_spike_train, ref_spike_train)
        t_start = 0
        for j, t in enumerate(spike_sync.x[1:]):
            if t == 1:
                t_end == 0.999
            else:
                t_end = t
            spike_sync_mat[i,int(t_start*time_scale):int(t_end*time_scale)] = spike_sync.y[j]
            t_start = t_end
        
    return spike_sync_mat


def evaluate_crps(encoder, flow_net, linear_transform,
                  device,
                  target_neuron,
                  important_index,
                  window_size,
                  time_resolution,
                  data_spike,
                  data_smooth,
                  stimuli,
                  filler=-1,
                  sigma=0.1,
                  scaling_factor=1,
                  smooth=True,
                  num_samples=2000):
    data = split_window_per_neuron_flow(data_spike, data_smooth, important_index,
                                        window_size, target_neuron, 
                                        time_resolution, filler,get_last_inf=False)
    
    data_target, data_source, data_smooth, time_conditional = data[0], data[1], data[2], data[3]
    data_target = torch.from_numpy(data_target).float().to(device)
    data_smooth = torch.from_numpy(data_smooth).float().to(device)
    data_source = torch.from_numpy(data_source).float().to(device)
    stimuli     = torch.from_numpy(stimuli).float().to(device)
    time = torch.from_numpy(time_conditional).float().to(device)
    stimuli_rep = stimuli.repeat(data_smooth.shape[0],1)
    
    if smooth:
        data_input = data_smooth
    else:
        data_input = data_source
    rnn_out, hidden = encoder(data_input) # get latent representation
            
    conditional = torch.cat([rnn_out, time], -1) # NF condition on context, neuron, and stimuli
    cond_repeat = torch.repeat_interleave(conditional, num_samples, 0)
    samples = flow_net.sample(cond_inputs=cond_repeat).reshape(conditional.shape[0],20,int(num_samples//20))/scaling_factor
    transpose_samples = torch.transpose(samples,0,1)
    crps = calculate_CRPS(data_target, transpose_samples)
    return crps
    
def calculate_CRPS(observed_sample, generated_sample):
    crps_list = []
    for i in range(generated_sample.shape[0]):
        first_term = torch.abs(generated_sample[i] - observed_sample).mean()
        gen_interleave = torch.repeat_interleave(generated_sample[i], generated_sample[i].shape[1],1)
        gen_repeat = generated_sample[i].repeat(1,generated_sample[i].shape[1])
        second_term = torch.abs(gen_interleave - gen_repeat).mean()*0.5
        crps_list.append((first_term - second_term).detach().cpu().numpy())
    return crps_list

def evaluate_likelihood_spike(encoder, flow_net, linear_transform,
                                device,
                                target_neuron,
                                important_index,
                                window_size,
                                time_resolution, 
                                data_spike, data_smooth, gen_spike_list, stimuli,
                                filler=-1,
                                sigma=0.1,
                                smooth=True):
    
    # True data likelihood
    data = split_window_per_neuron_flow(data_spike, data_smooth, important_index,
                                        window_size, target_neuron, 
                                        time_resolution, filler,get_last_inf=False)
    
    data_target, data_source, data_smooth, time_conditional = data[0], data[1], data[2], data[3]
    data_target = torch.from_numpy(data_target).float().to(device)
    data_smooth = torch.from_numpy(data_smooth).float().to(device)
    data_source = torch.from_numpy(data_source).float().to(device)
    stimuli     = torch.from_numpy(stimuli).float().to(device)
    time = torch.from_numpy(time_conditional).float().to(device)
    stimuli_rep = stimuli.repeat(data_smooth.shape[0],1)
    
    if smooth:
        data_input = data_smooth
    else:
        data_input = data_source
    rnn_out, hidden = encoder(data_input) # get latent representation
            
    #conditional = torch.cat([rnn_out, stimuli_rep, time], -1) # NF condition on context, neuron, and stimuli
    conditional = torch.cat([rnn_out, time], -1)
    #conditional = torch.cat([stimuli_rep, time], -1)
    likelihood_data = flow_net.log_probs(data_target, 
                                    cond_inputs=conditional).mean().item()
    
    likelihood_gen_list = []
    # Generated Data Likelihood 
    for gen in gen_spike_list:
        gen_smooth = gaussian_smoothing_spike(gen, time_resolution, sigma)
        data = split_window_per_neuron_flow(gen, gen_smooth, important_index,
                                        window_size, target_neuron, 
                                        time_resolution, filler,get_last_inf=False)
    
        data_target, data_source, data_smooth, time_conditional = data[0], data[1], data[2], data[3]
        data_target = torch.from_numpy(data_target).float().to(device)
        data_smooth = torch.from_numpy(data_smooth).float().to(device)
        data_source = torch.from_numpy(data_source).float().to(device)
        
        stimuli_rep = stimuli.repeat(data_smooth.shape[0],1)
        time = torch.from_numpy(time_conditional).float().to(device)
        if smooth:
            data_input = data_smooth
        else:
            data_input = data_source
        rnn_out, hidden = encoder(data_input) # get latent representation
                
        #conditional = torch.cat([rnn_out, stimuli_rep, time], -1) # NF condition on context, neuron, and stimuli
        conditional = torch.cat([rnn_out, time], -1)
        likelihood_gen = flow_net.log_probs(data_target, 
                                        cond_inputs=conditional).mean()
        likelihood_gen_list.append(likelihood_gen.item())

    return likelihood_data, likelihood_gen_list