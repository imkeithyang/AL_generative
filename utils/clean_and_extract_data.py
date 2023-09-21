import os
import json
import pandas as pd
import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt

def data_preprocessing(path, moth, behavioral_labels, duration, num_pulses=5):
    load_path = os.path.join(path, f'{moth}_prep.mat')
    data = loadmat(load_path)
    for key in ['__header__', '__version__', '__globals__', 'stim']:
        if key in data.keys():
            del data[key]
    
    info_path = os.path.join(path, f'timestamps_{moth}.csv')
    info = pd.read_csv(info_path, header=0)
    spike_train_data = {}
    spike_train_data.update({'label': []})
    spike_train_data.update({'stimuli': []})
    for stimuli in info.columns:
        if stimuli in behavioral_labels:
            target = 1
        else:
            target = 0
        for i in range(num_pulses):
            left = info[stimuli][i]
            right = info[stimuli][i]+duration
            spike_train_data['label'].append(target)
            spike_train_data['stimuli'].append(stimuli)
            for key in data.keys():
                if key not in spike_train_data.keys():
                    spike_train_data[key] = []
                spike_train_neuron = data[key]
                
                left = info[stimuli][i]-0.2
                right = info[stimuli][i]+duration
                if len(np.where(spike_train_neuron>left)[0]) == 0 or len(np.where(spike_train_neuron>right)[0]) == 0:
                    spikes = []
                else:
                    left_idx = np.where(spike_train_neuron>left)[0][0]
                    right_idx = np.where(spike_train_neuron>right)[0][0]
                    if left_idx==right_idx:
                        spikes = []
                    else:
                        spikes = [(x-left) for sublist in spike_train_neuron[left_idx:right_idx] for x in sublist]
                spike_train_data[key].append(spikes)
    spike_train_data_df = pd.DataFrame(data = spike_train_data)
    save_path = os.path.join(path, f'{moth}_cleaned.csv')
    spike_train_data_df.to_csv(save_path)

def gaussian_kernel_filtering(path, moth, duration, fs, sigma=0.05, plot = False):
    time= np.arange(0, duration, 1/fs)
    load_path = os.path.join(path, f'{moth}_cleaned.csv')
    data = pd.read_csv(load_path, index_col=0)
    gk_target = list(data['label'])
    gk_data = []
    del data['label']
    del data['stimuli']
    for i in range(data.shape[0]):
        feature = np.array([])
        for neuron in data.columns:
            gaussian_filtering_spike = np.zeros(len(time))
            neuron_spikes = json.loads(data[neuron][i])
            if len(neuron_spikes)==0:
                pass
            else:
                for spike in neuron_spikes:
                    gaussian_filtering_spike += np.exp(-1/2*(time-spike)**2/(sigma**2))
                if plot:
                    plt.plot(gaussian_filtering_spike)
                    plt.show()
                    plot = False
            two_feature = np.array([np.argmax(gaussian_filtering_spike), np.max(gaussian_filtering_spike)])
            feature = np.append(feature, gaussian_filtering_spike) 
            # feature = np.append(feature, two_feature) 
        gk_data.append(list(feature))
    return gk_data, gk_target

if __name__ == "__main__":
    path = "../ALdata"
    moth_names = ['070906', '070913', '070921', '070922', '070924_1', '070924_2', '071002']
    behavioral_labels = ['P9','P9_Ten','P9_Hund','P9_TenThous','P5','P4','P3']
    duration = 0.999
    fs = 1e4
    for m in moth_names:
        data_preprocessing(path, m, behavioral_labels, duration)
        data, target = gaussian_kernel_filtering(path, m, duration, fs, sigma=0.05)

