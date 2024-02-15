from utils import *
import pickle
import yaml
import copy
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import pandas as pd
import numpy as np
import json
from .get_data import read_moth


stimuli = ['P9_TenThous', 'M4', 'Bol', 'Ctl', 'DatExt', 'Far', 'Ger', 'Iso', 
           'Lin', 'M2', 'M3', 'M5', 'P9_Ten', 'M6', 'Mal', 'Myr', 'Ner', 'P3', 
           'P4', 'P5', 'P9', 'P9_Hund', 'Bea']

def analyze_deepAR(yaml_filepath, verbose=True):
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    if isinstance(cfg["data"]["target"], list):
        target_list = cfg["data"]["target"]
    else:
        target_list = [cfg["data"]["target"]]

    n_runs = cfg["n_runs"]
    
    important_threshold = cfg["important_threshold"] if "important_threshold" in cfg else 1.5
    important_index_list = []
    for target in target_list:
        cfg_temp = deepcopy(cfg)
        cfg_temp["data"]["target"] = target
        for stimuli_index in range(0,23):
            important_index = None
            # check if we have done previous preprocessing or not
            savepath, plot_savepath, net_savepath, exp = format_directory(cfg_temp, None, stimuli_index)
            important_index_file = os.path.join("/hpc/home/hy190/AL_generative",exp,'important_index_deepAR.pkl')
            if os.path.isfile(important_index_file) and os.path.getsize(important_index_file) > 0:
                with open(important_index_file, 'rb') as f:
                    important_dict = pickle.load(f)
                    loss_ratio = important_dict["loss_ratio"]
                    important_index = list(np.where(loss_ratio > important_threshold)[0])
                    if target not in set(important_index):
                        important_index.append(target)
                        important_index_list.append(important_index)
                f.close()
                if verbose:
                    print("Important Neuron For Target {}, Stimuli {}: {} ".format(target, 
                                                                                   stimuli_index, 
                                                                                   important_index))
    return important_index_list


def analyze_result(yaml_filepath, verbose=True, cond=False): 
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
        
    if isinstance(cfg["data"]["target"], list):
        target_list = cfg["data"]["target"]
    else:
        target_list = [cfg["data"]["target"]]

    cfg["use_component"] = ("use_component" in yaml_filepath)
    n_runs = cfg["n_runs"]
    count_all_target_reject = []
    
    if "PN" in yaml_filepath:
        neuron_type = "PN"
    elif "LN" in yaml_filepath:
        neuron_type = "LN"
    else:
        neuron_type = None
    
    for target in target_list:
        cfg_temp = copy.deepcopy(cfg)
        count_reject = np.array([])
        isi_stack = []
        spike_stack = []
        crps_stack = []
        pred = json.load(open("unlabeled_pred.json",'r'))
        _, neurons = read_moth(cfg_temp["data"]["path"], neuron_type=None, pred_label=pred, addtype=True)
        neuron=neurons[target]
        for run in range(n_runs):
            cfg_temp["data"]["target"] = target
            if cond:
                savepath, plot_savepath, net_savepath, exp = format_directory(cfg_temp, run, 
                                                                              neuron_type=neuron_type,
                                                                              neuron=target if not neuron_type else neuron[:-2])
            else:
                savepath, plot_savepath, net_savepath, exp = format_directory(cfg_temp, run, stimuli=0)
            try:
                with open(os.path.join(savepath,'test_stats.pkl'), 'rb') as f:
                    test_stats_run = pickle.load(f)
            except FileNotFoundError:
                scratch_path = "/scratch/hy190/AL_generative/"
                with open(os.path.join(scratch_path,savepath,'test_stats.pkl'), 'rb') as f:
                    test_stats_run = pickle.load(f)
                    
            data_likelihood_list = test_stats_run["data_likelihood_list"]
            gen_likelihood_list = test_stats_run["gen_likelihood_list"]
            crps_list = test_stats_run["crps_list"]
            isi_distance_list = test_stats_run["isi_distance_list"]
            spike_distance_list = test_stats_run["spike_distance_list"]
            
            crps_stack.append(np.array(crps_list).T)
            isi_stack.append(np.array(isi_distance_list).T)
            spike_stack.append(np.array(spike_distance_list).T)
            
            # Dealing with 0 variance likelihood 
            likelihood_lower = np.array(data_likelihood_list) < np.quantile(gen_likelihood_list, 0.025, axis=1)
            likelihood_higher = np.array(data_likelihood_list) > np.quantile(gen_likelihood_list, 0.975, axis=1)
            likelihood_outside_range = likelihood_lower+likelihood_higher
            zero_var_index = np.where(np.array(gen_likelihood_list).var(1) == 0)
            zero_var_likelihood_check = (np.array(data_likelihood_list)[zero_var_index] != \
                np.array(gen_likelihood_list)[:,0][zero_var_index])
            zero_var_outside_range = zero_var_index[0][zero_var_likelihood_check]
            likelihood_outside_range[zero_var_outside_range] = 1    
            
            count_reject = np.vstack([count_reject, likelihood_outside_range]) \
                if count_reject.size else likelihood_outside_range

        count_reject_all_run = count_reject.sum(0)
        mean_isi_stack = np.mean(isi_stack, 0)
        mean_spike_stack = np.mean(spike_stack, 0)
        mean_crps_stack = np.mean(crps_stack,0)
        crps_stack = np.array(crps_stack)
        isi_stack = np.array(isi_stack)
        spike_stack = np.array(spike_stack)
        if verbose:
            print("Neuron {} No. Rejected Emp-Likelihood by stimuli: {}, Reject rate: {}".format(target,
                                                                    count_reject_all_run.astype(int),
                                                                    np.round(np.mean(count_reject_all_run)/n_runs,2)))
            print("Neuron {} CRPS by stimuli: {}".format(target,
                                                      np.round(np.mean(mean_crps_stack,0),2)))
            print("Neuron {} ISI Distance by stimuli: {}".format(target,
                                                                np.round(np.mean(mean_isi_stack,0),2)))
            print("Neuron {} SPIKE Distance by stimuli: {}".format(target,
                                                                np.round(np.mean(mean_spike_stack,0),2)))
            count_all_target_reject.append(count_reject_all_run)
            print("Overall Rejection Rate: {}".format(np.mean(count_all_target_reject)/n_runs))
    
    fla_crps_stack = crps_stack.reshape(crps_stack.shape[0]*crps_stack.shape[1],-1)
    fla_isi_stack = isi_stack.reshape(isi_stack.shape[0]*isi_stack.shape[1],-1)
    fla_spike_stack = spike_stack.reshape(spike_stack.shape[0]*spike_stack.shape[1],-1)
    del test_stats_run
    torch.cuda.empty_cache()
    return fla_crps_stack, fla_isi_stack, fla_spike_stack, neurons[target]


def plot_result_bar(crps_all,
                    isi_all, 
                    spike_all,
                    target,method):
    method_list = []
    totlen = 0
    for i,met in enumerate(method):
        totlen += len(np.array(crps_all[i]).flatten())
        method_list += [met]*len(np.array(crps_all[i]).flatten())
    stimuli_list = stimuli*int(totlen/len(stimuli))
    crps_dict = {"CRPS": np.concatenate([np.array(crps_all[i]).flatten() for i in range(len(crps_all))]),
                 "Stimuli": stimuli_list,
                 "Method": method_list}
    method_list = []
    totlen = 0
    for i,met in enumerate(method):
        totlen += len(np.array(isi_all[i]).flatten())
        method_list += [met]*len(np.array(isi_all[i]).flatten())
    stimuli_list = stimuli*int(totlen/len(stimuli))
    isi_dict = {"ISI Distance": np.concatenate([np.array(isi_all[i]).flatten() for i in range(len(isi_all))]),
                 "Stimuli": stimuli_list,
                 "Method": method_list}
    method_list = []
    totlen = 0
    for i,met in enumerate(method):
        totlen += len(np.array(isi_all[i]).flatten())
        method_list += [met]*len(np.array(spike_all[i]).flatten())
    stimuli_list = stimuli*int(totlen/len(stimuli))
    spike_dict = {"SPIKE Distance": np.concatenate([np.array(spike_all[i]).flatten() for i in range(len(spike_all))]),
                 "Stimuli": stimuli_list,
                 "Method": method_list}
    
    fig, axes = plt.subplots(figsize=(24,5), ncols=3, nrows=1)
    sns.barplot(data = pd.DataFrame(crps_dict), x="Stimuli", y="CRPS", hue="Method", ax=axes[0],errorbar=('ci', 68))
    axes[0].set_yticks(axes[0].get_yticks()); axes[0].set_xticks(axes[0].get_xticks())
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60, ha='right')
    sns.barplot(data = pd.DataFrame(isi_dict), x="Stimuli", y="ISI Distance", hue="Method", ax=axes[1],errorbar=('ci', 68))
    axes[1].set_yticks(axes[1].get_yticks()); axes[1].set_xticks(axes[1].get_xticks())
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60, ha='right')
    sns.barplot(data = pd.DataFrame(spike_dict), x="Stimuli", y="SPIKE Distance", hue="Method", ax=axes[2],errorbar=('ci', 68))
    axes[2].set_yticks(axes[2].get_yticks()); axes[2].set_xticks(axes[2].get_xticks())
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=60, ha='right')
    
    fig.suptitle("Neuron {} Goodness of Fit".format(target))
    
def analyze_betai(yaml_filepath, cond=False, q = ['0-Bea', '0-Bol', '0-Ctl', '1-DatExt', '0-Far', '0-Ger', '0-Iso', '0-Lin', 
                 '0-M2', '0-M3', '0-M4', '0-M5', '0-M6', '0-Mal', '0-Myr', '0-Ner', 
                 '1-P3', '1-P4', '1-P5', '1-P9', '1-P9_Hund', '1-P9_Ten', '1-P9_TenThous'], addtype=False):
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
        
    cfg["data"]["use_component"] = ("use_component" in yaml_filepath)
    cfg_temp = copy.deepcopy(cfg)
    cfg_temp["data"]["target"] = cfg_temp["data"]["target"][0]
    n_runs = cfg_temp["n_runs"]
    time_scale = 10**cfg_temp["data"]["time_resolution"]
    if "PN" in yaml_filepath:
        neuron_type = "PN"
    elif "LN" in yaml_filepath:
        neuron_type = "LN"
    else:
        neuron_type = None
    pred = json.load(open("unlabeled_pred.json",'r'))
    _, neurons = read_moth(cfg_temp["data"]["path"], neuron_type=neuron_type, pred_label=pred, addtype=addtype)
    neurons.sort()
    target = cfg_temp["data"]["target"]
    neuron=neurons[target]
    betai_matrix_stack = []
    ensemble_stack = []
    pre_stim_ensemble_stack = []
    stim_ensemble_stack = []
    post_stim_ensemble_stack = []
    for run in range(n_runs):
        n_temp = neurons[cfg_temp["data"]["target"]][:-2] if addtype else neurons[cfg_temp["data"]["target"]]
        if neuron_type == "PN" and neurons[cfg_temp["data"]["target"]] not in set([
                                                                                   "S1U3","S2U1","S2U2",
                                                                                   "S3U4","S3U3"]):
            continue
        if cfg_temp["data"]["target"] >= len(neurons):
            return
        if cond:
                savepath, plot_savepath, net_savepath, exp = format_directory(cfg_temp, run, 
                                                                              neuron_type=neuron_type,
                                                                              neuron=target if not neuron_type else neuron[:-2])
        else:
                savepath, plot_savepath, net_savepath, exp = format_directory(cfg_temp, run, stimuli=0)
        try:
            with open(os.path.join(savepath,'test_stats.pkl'), 'rb') as f:
                    test_stats_run = pickle.load(f)
        except FileNotFoundError:
            scratch_path = "/scratch/hy190/AL_generative/"
            with open(os.path.join(scratch_path,savepath,'test_stats.pkl'), 'rb') as f:
                test_stats_run = pickle.load(f)
        
        betai_list = test_stats_run["betai_list"]
        time_list = test_stats_run["time_list"]
        spike_length = test_stats_run["data_emp"][0].shape[0]
        betai_matrix_list, pre_stim_ensemble_list, stim_ensemble_list, post_stim_ensemble_list =\
            build_beta_matrix(q, betai_list, time_list, time_scale, spike_length)
        betai_matrix_stack.append(betai_matrix_list)
        pre_stim_ensemble_stack.append(pre_stim_ensemble_list)
        stim_ensemble_stack.append(stim_ensemble_list)
        post_stim_ensemble_stack.append(post_stim_ensemble_list)
    try:
        del test_stats_run
    except:
        pass
    torch.cuda.empty_cache()
    return betai_matrix_stack, pre_stim_ensemble_stack, stim_ensemble_stack, post_stim_ensemble_stack, neurons

def build_beta_matrix(q, betai_list, time_list, time_scale, spike_length):
    betai_matrix_list = []
    ensemble_list = []
    pre_stim_ensemble_list = []
    stim_ensemble_list = []
    post_stim_ensemble_list = []
    for i, stimuli in enumerate(q):
        betai_matrix = np.zeros((betai_list[i].shape[1],spike_length))
        t_start = 0
        for j,t in enumerate(time_list[i][1:]):
            t_end = int(t*time_scale)
            betai_matrix[:,t_start:t_end] = betai_list[i][j].detach().cpu().numpy()
            t_start = t_end
        betai_matrix_list.append(betai_matrix)
        pre_stim_ensemble_list.append(betai_matrix[:,0:400].mean(1))
        stim_ensemble_list.append(betai_matrix[:,400:800].mean(1))
        post_stim_ensemble_list.append(betai_matrix[:,800:].mean(1))
        
    return betai_matrix_list, pre_stim_ensemble_list, stim_ensemble_list, post_stim_ensemble_list
            
            
def plot_ensemble(ensemble_list, method_list, target, q_labels,  neurons, section="During"):
    name_sort = np.argsort(neurons)
    stim_sort = np.argsort(q_labels)
    fig, axes = plt.subplots(ncols=len(ensemble_list), nrows=1, figsize=(3*len(ensemble_list), 5))
    
    if len(ensemble_list) == 1:
        axes.imshow(np.array(ensemble_list[0]).mean(0)[stim_sort,:][:,name_sort], aspect='auto')
        axes.set_xticks(list(range(len(neurons))))
        axes.set_xticklabels(np.array(neurons)[name_sort])
        axes.set_yticks(list(range(len(q_labels))))
        axes.set_yticklabels(np.array(q_labels)[stim_sort])
        axes.tick_params(axis='x', rotation=90)
    else:
        for i, ensemble in enumerate(ensemble_list):
            if np.array(ensemble_list[i]).shape[0] == 23:
                axes[i].imshow(np.array(ensemble_list[i]).mean(-1)[stim_sort,:][:,name_sort], aspect='auto')
            else:
                axes[i].imshow(np.array(ensemble_list[i]).mean(0)[stim_sort,:][:,name_sort], aspect='auto')
            if i == 0:
                axes[i].set_xticks(list(range(len(neurons))))
                axes[i].set_xticklabels(np.array(neurons)[name_sort])
                axes[i].set_yticks(list(range(len(q_labels))))
                axes[i].set_yticklabels(np.array(q_labels)[stim_sort])
                axes[i].tick_params(axis='x', rotation=90)
            else:
                axes[i].set_xticks(list(range(len(neurons))))
                axes[i].set_xticklabels(np.array(neurons)[name_sort])
                axes[i].set_yticklabels([])
                axes[i].tick_params(axis='x', rotation=90)
            axes[i].set_title(method_list[i])
    #plt.suptitle("{} Stimulus Neuron {} Spatio Attention".format(section, target))
    plt.tight_layout()