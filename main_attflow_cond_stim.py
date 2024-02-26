import yaml
import torch
import pickle
import copy
    
import os
from utils import *
from flow_att_cond_stim import *
from pathlib import Path

args = get_parser().parse_args()
if args.device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else: 
    device = torch.device(args.device)

yaml_filepath = args.filename
with open(yaml_filepath, 'r') as f:
    cfg = yaml.load(f, yaml.SafeLoader)


print("cfg:{}".format(cfg['data']['path']))

if args.path:
    cfg['data']['path'] = args.path

# Check if training needs neuron type
if "PN" in yaml_filepath:
    neuron_type = "PN"
elif "LN" in yaml_filepath:
    neuron_type = "LN"
else:
    neuron_type = None
print(f"neuron type: {neuron_type}")
# Different Attention/Flow Net structure from the unconditional attflow
net_yamlfilepath = Path(yaml_filepath).parent.parent
net_yamlfilepath = os.path.join(net_yamlfilepath, "sparse-attflow-net.yaml") if "sparse" in yaml_filepath else \
    os.path.join(net_yamlfilepath, "attflow-net.yaml")
    
with open(net_yamlfilepath, 'r') as f:
    cfg_net = yaml.load(f, yaml.SafeLoader)

cfg["att_encoder"] = cfg_net["att_encoder"]    
cfg["flow_net"] = cfg_net["flow_net"]
cfg["data"]["batch_size"] = cfg_net["batch_size"]
cfg["data"]["use_component"] = ("use_component" in yaml_filepath)
n_runs = cfg['n_runs']
n_tries = cfg['n_tries']


if isinstance(cfg["data"]["target"], list):
    target_list = cfg["data"]["target"]
else:
    target_list = [cfg["data"]["target"]]
    
for target in target_list:
    print("****************** Training Target {} ******************".format(target))
    cfg_temp = copy.deepcopy(cfg)
    cfg_temp["data"]["target"] = target
    
    important_index = None
    
    for run in range(0,n_runs):
        q_temp = []
        data_gen = []
        data_emp = []
        crps_list = []
        betai_list = []
        alphai_list = []
        spike_sync_list = []
        time_list = []
        data_likelihood_list = []
        gen_likelihood_list = []
        spike_train_list = []
        isi_distance_list = []
        spike_distance_list = []
        
        smooth=True
        initialized, test_loader, data_spike, data_smooth, q,neurons = setup_att_flow(cfg_temp, 
                                                                            important_index, 
                                                                            device, run=run,
                                                                            neuron_type=neuron_type)
        savepath, plot_savepath, net_savepath,exp = format_directory(cfg_temp, run, 
                                                                     neuron_type=neuron_type,
                                                                     neuron=neurons[cfg_temp['data']['target']])
        make_directory(exp, savepath, plot_savepath, net_savepath)
        
        initialized["paths"] = (savepath, plot_savepath, net_savepath)
        initialized["device"] = device
        initialized["smooth"] = smooth
        all_stats, best_epoch = train_att_flow(**initialized)
        plot_loss(all_stats, initialized["n_epochs"], savepath)
        encoder_best = initialized["encoder"]
        encoder_best.load_state_dict(torch.load(net_savepath + "/encoder.pt"))
        encoder_best.eval()
        
        flow_net_best = initialized["flow_net"]
        flow_net_best.load_state_dict(torch.load(net_savepath + "/flow_net.pt"))
        flow_net_best.eval()
        
        linear_transform_best = None
        test_stats = validate_att_flow(encoder_best, flow_net_best, linear_transform_best, 
                                            test_loader, device)
        
        window_size     = initialized["window_size"]
        n_neurons       = initialized["n_neurons"]
        filler          = initialized["filler"]
        time_resolution = initialized["time_resolution"]
        target_neuron   = initialized["target_neuron"]
        scaling_factor  = initialized["scaling_factor"]
        sigma           = initialized["sigma"]
        important_index = initialized["important_index"]
        stim_name       = initialized["stim_name"]
        neuron_names = []

        for stimuli, d_spike, d_smooth in zip(q, data_spike, data_smooth):
            spike_train = generate_spike_train_att_flow(encoder_best, flow_net_best,linear_transform_best, 
                                                        device,
                                                        target_neuron,
                                                        important_index,
                                                        window_size, 
                                                        n_neurons, 
                                                        torch.from_numpy(stimuli).float(), 
                                                        time_resolution, 
                                                        sigma,
                                                        torch.from_numpy(d_spike).float(), 
                                                        torch.from_numpy(d_smooth).float(),
                                                        scaling_factor=scaling_factor,
                                                        filler=filler, 
                                                        smooth=smooth,
                                                        num_of_spike_train=5)
            spike_train_list.append(spike_train)
            data_emp.append(d_spike)
            data_gen.append(spike_train[0])
            q_temp.append(stimuli)
            isi_dist, spike_dist = evaluate_spike_distance(d_spike, spike_train, target_neuron, time_resolution)
            isi_distance_list.append(isi_dist)
            spike_distance_list.append(spike_dist)
            likelihood_data, likelihood_gen = evaluate_likelihood_spike(encoder_best, flow_net_best, linear_transform_best,
                                                                        device,
                                                                        target_neuron,
                                                                        important_index,
                                                                        window_size,
                                                                        time_resolution,
                                                                        d_spike,d_smooth,
                                                                        spike_train,
                                                                        stimuli,
                                                                        filler=filler,
                                                                        sigma=sigma,
                                                                        smooth=smooth)
            data_likelihood_list.append(likelihood_data)
            gen_likelihood_list.append(likelihood_gen)
            
            crps = evaluate_crps(encoder_best, flow_net_best,linear_transform_best,
                            device,
                            target_neuron,
                            important_index,
                            window_size,
                            time_resolution,
                            d_spike,
                            d_smooth,
                            stimuli,
                            filler=filler,
                            sigma=sigma,
                            scaling_factor=scaling_factor,
                            smooth=smooth,
                            num_samples=2000)
            crps_list.append(crps)
            
            time, betai = evaluate_betai(encoder_best, device, 
                                        d_spike, d_smooth, stimuli, important_index,sigma,
                                        window_size, target_neuron, 
                                        time_resolution, filler, smooth=smooth)
            time, alphai = evaluate_alphai(encoder_best, device, 
                                        d_spike, d_smooth, stimuli, important_index,sigma,
                                        window_size, target_neuron, 
                                        time_resolution, filler, smooth=smooth)
            spike_sync_mat = evaluate_spikesync_unit_pairs(d_spike, target_neuron, time_resolution)
            alphai_list.append(alphai)
            betai_list.append(betai)
            time_list.append(time)
            spike_sync_list.append(spike_sync_mat)
                
        # Dealing with 0 variance likelihood 
        likelihood_lower = np.array(data_likelihood_list) < np.quantile(gen_likelihood_list, 0.025, axis=1)
        likelihood_higher = np.array(data_likelihood_list) > np.quantile(gen_likelihood_list, 0.975, axis=1)
        likelihood_outside_range = likelihood_lower+likelihood_higher
        zero_var_index = np.where(np.array(gen_likelihood_list).var(1) == 0)
        zero_var_likelihood_check = (np.array(data_likelihood_list)[zero_var_index] != \
            np.array(gen_likelihood_list)[:,0][zero_var_index])
        zero_var_outside_range = zero_var_index[0][zero_var_likelihood_check]
        likelihood_outside_range[zero_var_outside_range] = 1
        
        print("Target {}, Run {} Out of all {} stimuli, Rejected {} Emp Likelihood-test".format(target,
                                                                                                run,
                                                                                                len(q), 
                                            np.sum(likelihood_outside_range).astype(int)))   
        print("CRPS by Stimuli: {}".format(np.round(crps_list, 3)),"\n")
        print("ISI-dist by Stimuli: {}".format(np.round(np.mean(isi_distance_list, axis=1),3)),"\n")
        print("SPIKE-dist by Stimuli: {}".format(np.round(np.mean(spike_distance_list, axis=1),3)),"\n")
        
        plot_spike_compare(data_emp, 
                        data_gen, 
                        important_index,
                        savepath, 
                        "test",
                        q, 
                        target=target_neuron,
                        data_likelihood_list = data_likelihood_list,
                        gen_likelihood_list = gen_likelihood_list, stim_name=stim_name)
        
        spike_length = data_spike[0].shape[0]
        plot_betai_compare(time_list, betai_list, spike_sync_list, spike_length, time_resolution,
                   savepath, "test", q_temp, target, stim_name=stim_name,neuron_names = neuron_names)
        plot_spatiotemporal_compare(time_list, betai_list, alphai_list, window_size, 
                                    spike_length, time_resolution,
                   savepath, "test", q_temp, target, stim_name=stim_name,neuron_names = neuron_names)
        
        test_stats = {"test_stats":test_stats,"crps_list":crps_list,
                  "data_emp":np.array(data_emp), "data_gen":np.array(data_gen),
                  "data_likelihood_list":data_likelihood_list,
                  "gen_likelihood_list":gen_likelihood_list,
                  "spike_train_list":spike_train_list, 
                  "isi_distance_list":isi_distance_list, 
                  "spike_distance_list":spike_distance_list,
                  "time_list": time_list,
                  "betai_list": betai_list,
                  "alphai_list": alphai_list}
        
        with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
            pickle.dump(test_stats, f)
            f.close()
        with open(os.path.join(savepath,'saved_stats.pkl'), 'wb') as f:
            pickle.dump(all_stats, f)
            f.close()