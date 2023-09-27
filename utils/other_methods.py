#from result import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.io import loadmat
from get_data import *

stimuli = ['0-BEA', '0-BOL', '0-Ctl', '1-DatExt', '0-FAR', '0-GER', '0-ISO', '0-LIN', 
           '0-M2', '0-M3', '0-M4', '0-M5', '0-M6', '0-MAL', '0-MYR', '0-NER', 
           '1-P3', '1-P4', '1-P5', '1-P9', '1-P9_Hund', '1-P9_Ten', '1-P9_TenThous']		

mixture_dict = {"1-P9":["0-BEA","0-BOL","0-MAL","0-MYR","0-LIN","0-NER","0-GER","0-ISO","0-FAR"],
                "1-P5":["0-BEA","0-BOL","0-LIN","0-NER","0-GER"],
                "1-P4":["0-BEA","0-BOL","0-LIN","0-NER"],
                "1-P3":["0-BEA","0-BOL","0-LIN"],
                "0-M6":["0-MAL","0-MYR","0-NER","0-GER","0-ISO","0-FAR"],
                "0-M5":["0-MAL","0-MYR","0-GER","0-ISO","0-FAR"],
                "0-M4":["0-MAL","0-MYR","0-ISO","0-FAR"],
                "0-M3":["0-MAL","0-ISO","0-FAR"],
                "0-M2":["0-BEA","0-BOL"]}


def exponential_smoothing_spike(data_concat, time_resolution, tau=0.05):
    time_scale = 10**time_resolution
    tot_time = data_concat.shape[0]/time_scale
    data_concat_smooth = []
    for n in range(data_concat.shape[-1]):
        cat_temp = np.zeros(data_concat[:,n].shape).reshape(-1,1)
        spike_time_unscaled = np.nonzero(data_concat[:,n])[0]
        target_time = np.diff(spike_time_unscaled)/time_scale
        if spike_time_unscaled.size != 0:
            for t in spike_time_unscaled:
                temp = np.exp(-((np.arange(0,tot_time,1/time_scale)-t/time_scale)/tau)).reshape(-1,1)
                temp[0:t] = 0
                cat_temp += temp
        data_concat_smooth.append(cat_temp)
    data_concat_smooth = np.concatenate(data_concat_smooth, 1)
    return data_concat_smooth

def similarity_measure(data_concat_smooth):
    mat = np.zeros((data_concat_smooth.shape[1],data_concat_smooth.shape[1]))
    for i in range(data_concat_smooth.shape[1]):
        for j in range(i+1, data_concat_smooth.shape[1]):
            top = np.dot(data_concat_smooth[:,i], data_concat_smooth[:,j])
            bot = np.linalg.norm(data_concat_smooth[:,i])*np.linalg.norm(data_concat_smooth[:,j])
            if bot == 0:
                continue
            mat[i,j] = top/bot
            
    return mat

def similarity_measure_shuffle(data_concat_smooth, data_shuffle):
    mat = np.zeros((data_concat_smooth.shape[1],data_concat_smooth.shape[1]))
    for i in range(data_concat_smooth.shape[1]):
        for j in range(i+1, data_shuffle.shape[1]):
            top = np.dot(data_concat_smooth[:,i], data_shuffle[:,j])
            bot = np.linalg.norm(data_concat_smooth[:,i])*np.linalg.norm(data_shuffle[:,j])
            if bot == 0:
                continue
            mat[i,j] = top/bot
            
    return mat

            