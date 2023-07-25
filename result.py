from utils import *
import pickle
import yaml
import copy
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import pandas as pd
import numpy as np
from scipy.signal import correlate, correlation_lags
import json


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


def analyze_noncond_stim(yaml_filepath, verbose=True): 
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
        
    if isinstance(cfg["data"]["target"], list):
        target_list = cfg["data"]["target"]
    else:
        target_list = [cfg["data"]["target"]]

    n_runs = cfg["n_runs"]
    count_all_target_reject = []
    
    for target in target_list:
        cfg_temp = copy.deepcopy(cfg)
        count_reject = np.array([])
        isi_stack = []
        spike_stack = []
        crps_stack = []
        for run in range(n_runs):
            cfg_temp["data"]["target"] = target
            savepath, plot_savepath, net_savepath, exp = format_directory(cfg_temp, run, stimuli=0)
            with open(os.path.join(savepath,'test_stats.pkl'), 'rb') as f:
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
    return fla_crps_stack, fla_isi_stack, fla_spike_stack
    
def analyze_cond_stim(yaml_filepath, target, verbose=True): 
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
    cfg_temp = copy.deepcopy(cfg)
    cfg_temp["data"]["target"] = target
    n_runs = cfg_temp["n_runs"]
    
    count_reject = np.array([])
    isi_stack = []
    spike_stack = []
    crps_stack = []
    for run in range(n_runs):
        savepath, plot_savepath, net_savepath, exp = format_directory(cfg_temp, run)
        with open(os.path.join(savepath,'test_stats.pkl'), 'rb') as f:
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
    mean_crps_stack = np.mean(crps_stack, 0)
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
    return fla_crps_stack, fla_isi_stack, fla_spike_stack



def plot_result_bar(crps_all,
                    isi_all, 
                    spike_all,
                    target,method):
    method_list = []
    for met in method:
        method_list += [met]*len(np.array(crps_all[0]).flatten())
    stimuli_list = stimuli*len(crps_all[0])*len(crps_all)
    crps_dict = {"CRPS": np.concatenate([np.array(crps_all[i]).flatten() for i in range(len(crps_all))]),
                 "Stimuli": stimuli_list,
                 "Method": method_list}
    
    method_list = []
    for met in method:
        method_list += [met]*len(np.array(isi_all[0]).flatten())
    stimuli_list = stimuli*len(isi_all[0])*len(isi_all)
    isi_dict = {"ISI Distance": np.concatenate([np.array(isi_all[i]).flatten() for i in range(len(isi_all))]),
                 "Stimuli": stimuli_list,
                 "Method": method_list}
    
    method_list = []
    for met in method:
        method_list += [met]*len(np.array(spike_all[0]).flatten())
    stimuli_list = stimuli*len(spike_all[0])*len(spike_all)
    spike_dict = {"SPIKE Distance": np.concatenate([np.array(spike_all[i]).flatten() for i in range(len(spike_all))]),
                 "Stimuli": stimuli_list,
                 "Method": method_list}
    
    fig, axes = plt.subplots(figsize=(24,5), ncols=3, nrows=1)
    sns.barplot(data = pd.DataFrame(crps_dict), x="Stimuli", y="CRPS", hue="Method", ax=axes[0],ci=68)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=60, ha='right')
    sns.barplot(data = pd.DataFrame(isi_dict), x="Stimuli", y="ISI Distance", hue="Method", ax=axes[1],ci=68)
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=60, ha='right')
    sns.barplot(data = pd.DataFrame(spike_dict), x="Stimuli", y="SPIKE Distance", hue="Method", ax=axes[2],ci=68)
    axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=60, ha='right')
    
    fig.suptitle("Neuron {} Goodness of Fit".format(target))
    
    
def cross_correlation(path="/hpc/group/tarokhlab/hy190/data/AL/ALdata/070921_cleaned.csv"):

    df = pd.read_csv(path, index_col=0)
    neurons = list(df.keys())[2:]
    for i in range(df.shape[0]):
        for j in neurons:
            df.at[i,j] = np.round(json.loads(df.iloc[i][j]), 3)

    all_stimuli_count = df.value_counts("stimuli").to_dict()
    num_stimuli = len(all_stimuli_count)
    bins = np.arange(0,1,0.005)
    SI_index_by_stimuli = []
    mean_SI_index_by_stimuli = []
    for s_index, s in enumerate(all_stimuli_count):
        SI_index_list = []
        for run in range(all_stimuli_count[s]):
            # SE Raw
            se_raw = np.zeros((len(neurons), len(neurons)))
            se_shuffle = np.zeros((len(neurons), len(neurons)))
            N1_plus_N2 = np.zeros((len(neurons), len(neurons)))
            data_stimuli = df[df["stimuli"] == s]
            data_concat = np.zeros((1000,len(neurons)))
            for j, neuron_1 in enumerate(neurons):
                neuron_tar = data_stimuli.iloc[run][neuron_1]
                neuron_tar_low = neuron_tar - 0.01
                neuron_tar_high = neuron_tar + 0.01
                
                neuron_tar_low = neuron_tar_low.reshape(1,-1)
                neuron_tar_high = neuron_tar_high.reshape(1,-1)
                for k, neuron_2 in enumerate(neurons):
                    # SE Raw
                    neuron_ref = data_stimuli.iloc[run][neuron_1]
                    neuron_ref = neuron_ref.reshape(-1,1)
                    low_leq = np.less_equal(neuron_tar_low, neuron_ref)
                    high_geq = np.greater_equal(neuron_tar_high, neuron_ref)
                    
                    within_range = low_leq*high_geq
                    se_raw[j,k] = within_range.sum()
            #        # N1 + N2
            #        N1_plus_N2[j,k] = neuron_tar.sum() + neuron_ref.sum()
            #        # SE Shuffle
            #        for shift_predictor_run in range(all_stimuli_count[s]):
            #            if shift_predictor_run == run:
            #                continue
            #            data_concat_shuffle = np.zeros((1000,len(neurons)))
            #            for i, neuron in enumerate(neurons):
            #                data_concat_shuffle[(1000*data_stimuli.iloc[shift_predictor_run][neuron]).astype(int),i] = 1
            #                
            #            neuron_ref_shuffle = data_concat_shuffle[:,k]
            #            cross_corr_shuffle = correlate(neuron_ref_shuffle, neuron_tar)
            #            cross_cor_lags = correlation_lags(len(neuron_ref_shuffle), len(neuron_tar))
            #            se_shuffle[j,k] += cross_corr_shuffle[np.where((cross_cor_lags <= 5) & (cross_cor_lags >= -5))].sum()
                                
            se_shuffle = se_shuffle/(all_stimuli_count[s] - 1)
            SI_index = (se_raw - se_shuffle)/N1_plus_N2
            SI_index_list.append(SI_index) 
        mean_SI_index_by_stimuli.append(np.mean(SI_index_list, 0))
        SI_index_by_stimuli.append(SI_index_list)


#target = 0
#rnn_crps_stack, rnn_isi_stack, rnn_spike_stack = analyze_conditionalflow("config/rnnflow/rnnflow-{}.yaml".format(target), 
#                                                                         verbose=False)
#att_crps_stack, att_isi_stack, att_spike_stack = analyze_attentionflow(yaml_filepath="config/attflow.yaml", 
#                                                                       target=target, 
#                                                                       verbose=False)
#plot_result_bar([rnn_crps_stack, att_crps_stack],
#                [rnn_isi_stack, att_isi_stack],
#                [rnn_spike_stack, att_spike_stack],
#                target, method=["rnnflow", "attflow"])


cross_correlation()
#for i, neuron in enumerate(neurons):
#    data_concat[(1000*data_stimuli.iloc[run][neuron]).astype(int),i] = 1
#    
#for j, neuron in enumerate(neurons):
#    neuron_tar = data_concat[:,j]
#    for k, neuron in enumerate(neurons):
#        # SE Raw
#        neuron_ref = data_concat[:,k]
#        cross_corr = correlate(neuron_ref, neuron_tar)
#        cross_cor_lags = correlation_lags(len(neuron_ref), len(neuron_tar))
#        se_raw[j,k] = cross_corr[np.where((cross_cor_lags <= 5) & (cross_cor_lags >= -5))].sum()
#        # N1 + N2
#        N1_plus_N2[j,k] = neuron_tar.sum() + neuron_ref.sum()
#        # SE Shuffle
#        for shift_predictor_run in range(all_stimuli_count[s]):
#            if shift_predictor_run == run:
#                continue
#            data_concat_shuffle = np.zeros((1000,len(neurons)))
#            for i, neuron in enumerate(neurons):
#                data_concat_shuffle[(1000*data_stimuli.iloc[shift_predictor_run][neuron]).astype(int),i] = 1
#                
#            neuron_ref_shuffle = data_concat_shuffle[:,k]
#            cross_corr_shuffle = correlate(neuron_ref_shuffle, neuron_tar)
#            cross_cor_lags = correlation_lags(len(neuron_ref_shuffle), len(neuron_tar))
#            se_shuffle[j,k] += cross_corr_shuffle[np.where((cross_cor_lags <= 5) & (cross_cor_lags >= -5))].sum()