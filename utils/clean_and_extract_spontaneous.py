import os
import json
import pandas as pd
import numpy as np 
from scipy.io import loadmat
import matplotlib.pyplot as plt

def data_preprocessing(path, moth, behavioral_labels, duration, num_pulses=1):
    load_path = os.path.join(path, f'{moth}_prep.mat')
    data = loadmat(load_path)
    for key in ['__header__', '__version__', '__globals__', 'stim']:
        if key in data.keys():
            del data[key]
    
    info_path = os.path.join(path,f'timestamps_{moth}.csv')
    info = pd.read_csv(info_path, header=0)
    spike_train_data = {}
    spike_train_data.update({'label': []})
    spike_train_data.update({'stimuli': []})
    
    # stimuli = info.columns[np.argmin(info.iloc[0])]

    for stimuli in info.columns:
        if stimuli in behavioral_labels:
            target = 1
        else:
            target = 0
        for i in range(num_pulses):
            # left = info[stimuli][i]
            # right = info[stimuli][i]+duration
            spike_train_data['label'].append(target)
            spike_train_data['stimuli'].append(stimuli)
            for key in data.keys():
                if key not in spike_train_data.keys():
                    spike_train_data[key] = []
                spike_train_neuron = data[key]
                
                left = info[stimuli].iloc[0] - duration
                right = info[stimuli].iloc[0]
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
    save_path = os.path.join(path, f'{moth}_spontaneous.csv')
    spike_train_data_df.to_csv(save_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Data Preprocessing', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--path', type = str, default = "/hpc/group/tarokhlab/hy190/data/AL/ALdata",
                        help='data path')
    parser.add_argument('--pre_stim', dest='pre_stim', action='store_true', 
                        help='process pre stimulation or not')
    args = parser.parse_args()
    
    # moth_names = ['070906', '070913', '070921', '070922', '070924_1', '070924_2', '071002']
    moth_names = ['04021024']
    # behavioral_labels = ['P9','P9_Ten','P9_Hund','P9_TenThous','P5','P4','P3', 'DatExt']
    # behavioral_labels = ['Datura Mix ','Benzaldehye','Mineral oil', 'DAT+10X BEA (incr) ']
    behavioral_labels = ['TBD']

    duration = 4.999
    fs = 1e4
    for m in moth_names:
        print(args.pre_stim)
        data_preprocessing(args.path, m, behavioral_labels, duration)

        