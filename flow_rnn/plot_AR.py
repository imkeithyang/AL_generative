import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_spike(data_concat, neurons, plot_savepath, epoch, q):
    fig, ax = plt.subplots(figsize=(10,1*len(neurons)), nrows=len(neurons))
    for i, neuron in enumerate(neurons):
        if data_concat[:,i].sum() != 0:
            ax[i].plot(data_concat[:,i])
            ax[i].set_title(neuron.item(), loc="left")
    plt.tight_layout()
    plt.savefig("{}/{}-{}.png".format(plot_savepath, epoch, q))
    plt.close()
    
def plot_spike_compare(data_concat_true, data_concat_gen, important_neurons, 
                       plot_savepath, epoch, q, target, 
                       data_likelihood_list = None, gen_likelihood_list=None):
    fig, ax = plt.subplots(figsize=(10*2,1.5*len(q)), nrows=len(q), ncols=2 if data_likelihood_list is None else 3)
    if len(q) > 1:
        for i, stimuli in enumerate(q):
            sti = np.where(stimuli == 1)[1][0]
            ax[i][0].plot(data_concat_true[i][:,target])
            ax[i][0].set_title(target, loc="left")
            ax[i][0].set_title("Stimuli {} Empirical Neuron {}".format(sti, target), loc="left")

            ax[i][1].plot(data_concat_gen[i][:,target])
            ax[i][1].set_title("Stimuli {} Gen Neuron {}".format(sti, target), loc="left")
                
            if "pre_stimuli" in plot_savepath:
                ax[i][0].axvline(x=100, linestyle="--", color="black")
                ax[i][1].axvline(x=100, linestyle="--", color="black")
    else:
        sti = np.where(q[0] == 1)[1][0]
        ax[0].plot(data_concat_true[0][:,target])
        ax[0].set_title(target, loc="left")
        ax[0].set_title("Stimuli {} Empirical Neuron {}".format(sti, target), loc="left")
        ax[1].plot(data_concat_gen[0][:,target])
        ax[1].set_title("Stimuli {} Gen Neuron {}".format(sti, target), loc="left")
            
    if data_likelihood_list is not None:
        for i, stimuli in enumerate(q):
            sti = np.where(stimuli == 1)[1][0]
            if data_likelihood_list[i] < np.quantile(gen_likelihood_list[i], 0.025) or \
               data_likelihood_list[i] > np.quantile(gen_likelihood_list[i], 0.975):
                if len(q) > 1:
                    sns.kdeplot(x=gen_likelihood_list[i], ax=ax[i][-1], fill=True, alpha=0.3, color="red")
                else:
                    sns.kdeplot(x=gen_likelihood_list[i], ax=ax[-1], fill=True, alpha=0.3, color="red")
            else:
                if len(q) > 1:
                    sns.kdeplot(x=gen_likelihood_list[i], ax=ax[i][-1], fill=True, alpha=0.3, color="blue")
                else:
                    sns.kdeplot(x=gen_likelihood_list[i], ax=ax[-1], fill=True, alpha=0.3, color="blue")
            if len(q) > 1:
                ax[i][-1].axvline(x=data_likelihood_list[i], color="black")
                ax[i][-1].set_title("Likelihood Density".format(sti, target), loc="left")
            else:
                ax[-1].axvline(x=data_likelihood_list[i], color="black")
                ax[-1].set_title("Likelihood Density".format(sti, target), loc="left")

            
    plt.tight_layout()
    plt.savefig("{}/compare-{}-all-stimuli.png".format(plot_savepath, epoch))
    plt.close()
    
    
    
    
def plot_betai_compare(betai_list, spike_sync_list, spike_length, time_resolution,
                       plot_savepath, epoch, q, target):
    time_scale = 10**time_resolution
    fig, ax = plt.subplots(figsize=(14,3*len(q)), nrows=len(q), ncols=2)
    if len(q) > 1:
        for i, stimuli in enumerate(q):
            sti = np.where(stimuli == 1)[1][0]
            betai_matrix = np.zeros((betai_list[i].shape[0],spike_length))
            betai_matrix[:,:] = np.expand_dims(betai_list[i],1)
                
            ax[i][0].imshow(betai_matrix, aspect='auto')
            ax[i][0].set_yticks(list(range(betai_list[i].shape[0])))
            ax[i][0].set_title("Stimuli {} Neuron {} Beta Importance".format(sti, target), loc="left")
            
            ax[i][1].imshow(spike_sync_list[i], aspect='auto')
            ax[i][1].set_yticks(list(range(betai_list[i].shape[0])))
            ax[i][1].set_title("Stimuli {} Neuron {} SPIKE Sync".format(sti, target), loc="left")
            
    plt.tight_layout()
    plt.savefig("{}/betai-{}-all-stimuli.png".format(plot_savepath, epoch))
    plt.close()