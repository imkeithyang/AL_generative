import yaml
import torch
import pickle
import copy

from utils import *
from flow_rnn import *

args = get_parser().parse_args()
if args.device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else: 
    device = torch.device(args.device)
    
yaml_filepath = args.filename
with open(yaml_filepath, 'r') as f:
    cfg = yaml.load(f, yaml.SafeLoader)

cfg["data"]["use_component"] = ("use_component" in yaml_filepath)
n_runs = 5
n_tries = 1
    
if isinstance(cfg["data"]["target"], list):
    target_list = cfg["data"]["target"]
else:
    target_list = [cfg["data"]["target"]]
    
for target in target_list:
    print("****************** Taining Target {} ******************".format(target))
    cfg_temp = copy.deepcopy(cfg)
    cfg_temp["data"]["target"] = target
    
    important_threshold = cfg_temp["important_threshold"] if "important_threshold" in cfg_temp else 1.5
    ar_yaml_filepath = "AL_generative/config/deepAR/deepAR-{}.yaml".format(target) if os.getcwd().split("/")[-1] == "hy190" else \
        "config/deepAR/deepAR-{}.yaml".format(target)
    with open(ar_yaml_filepath, 'r') as f:
        cfg_deepAR = yaml.load(f, yaml.SafeLoader)
    cfg_deepAR["data"]["target"] = target
    for run in range(0,n_runs):
        
        q_temp = []
        data_emp = []
        data_gen = []
        crps_list = []
        betai_list = []
        data_likelihood_list = []
        gen_likelihood_list = []
        spike_train_list = []
        isi_distance_list = []
        spike_distance_list = []
        spike_sync_list = []
        
        for stimuli_index in range(0,23):
            important_index = None
            # check if we have done previous preprocessing or not
            savepath, plot_savepath, net_savepath, exp = format_directory(cfg_deepAR, None, stimuli_index)
            important_index_file = os.path.join("/hpc/home/hy190/AL_generative",exp,'important_index_deepAR.pkl')
            if os.path.isfile(important_index_file) and os.path.getsize(important_index_file) > 0:
                with open(important_index_file, 'rb') as f:
                    important_dict = pickle.load(f)
                    loss_ratio = important_dict["loss_ratio"]
                    print(np.round(important_dict["loss"],2))
                    print(np.round(loss_ratio,1))
                    important_index = list(np.where(loss_ratio > important_threshold)[0])
                    if target not in set(important_index):
                        important_index.append(target)
                f.close()
                print("Found Existing Index For Target {}, Stimuli {}: {} ".format(target, stimuli_index, important_index))
                loss_ratio[loss_ratio <= important_threshold] = 0
                betai_list.append(loss_ratio)
            else:
                print("Important Index For Target {}, Stimuli {} Not Found...Preprocessing".format(target, stimuli_index))
                exit()
            
            # Training Normalizing Flow
            smooth=True
            savepath, plot_savepath, net_savepath,exp = format_directory(cfg_temp, run, stimuli_index)
            make_directory(exp, savepath, plot_savepath, net_savepath)
            initialized, test_loader, data_spike, data_smooth, q = setup_conditional_flow(cfg_temp, 
                                                                                        important_index, 
                                                                                        device, run=run,seed=run,
                                                                                        stimuli_index = stimuli_index)
            initialized["paths"] = (savepath, plot_savepath, net_savepath)
            initialized["device"] = device
            initialized["smooth"] = smooth
            all_stats, best_epoch = train_conditional_flow(**initialized)
            
            encoder_best = initialized["encoder"]
            encoder_best.load_state_dict(torch.load(net_savepath + "/encoder.pt"))
            encoder_best.eval()
            
            flow_net_best = initialized["flow_net"]
            flow_net_best.load_state_dict(torch.load(net_savepath + "/flow_net.pt"))
            flow_net_best.eval()
            
            linear_transform_best = None
            
            test_stats = validate_conditional_flow(encoder_best, flow_net_best, linear_transform_best, 
                                                test_loader, device)
            
            window_size     = initialized["window_size"]
            n_neurons       = initialized["n_neurons"]
            filler          = initialized["filler"]
            time_resolution = initialized["time_resolution"]
            target_neuron   = initialized["target_neuron"]
            scaling_factor  = initialized["scaling_factor"]
            sigma           = initialized["sigma"]
            
            
            for stimuli, d_spike, d_smooth in zip(q, data_spike, data_smooth):
                spike_train = generate_spike_train_conditonal_flow(encoder_best, flow_net_best,linear_transform_best, 
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
                                                                num_of_spike_train=100)
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
                spike_sync_mat = evaluate_spikesync_unit_pairs(d_spike, target_neuron, time_resolution)
                spike_sync_list.append(spike_sync_mat)
                
        # Evaluation of all stimuli 
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
                                                                                                len(q_temp), 
                                            np.sum(likelihood_outside_range).astype(int)))   
        print("CRPS by Stimuli: {}".format(np.round(crps_list, 3)),"\n")
        print("ISI-dist by Stimuli: {}".format(np.round(np.mean(isi_distance_list, axis=1),3)),"\n")
        print("SPIKE-dist by Stimuli: {}".format(np.round(np.mean(spike_distance_list, axis=1),3)),"\n")
        plot_spike_compare(data_emp, 
                           data_gen, 
                           important_index,
                           savepath, 
                           "test",
                           q_temp, 
                           target=target_neuron,
                           data_likelihood_list = data_likelihood_list,
                           gen_likelihood_list = gen_likelihood_list)
        spike_length = data_spike[0].shape[0]
        plot_betai_compare_rnn(betai_list, spike_sync_list, spike_length, time_resolution,
                       savepath, "test", q_temp, target)
        
        test_stats = {"test_stats":test_stats,"crps_list":crps_list,
                      "data_emp":np.array(data_emp), "data_gen":np.array(data_gen),
                      "data_likelihood_list":data_likelihood_list,
                      "gen_likelihood_list":gen_likelihood_list,
                      "spike_train_list":spike_train_list, 
                      "isi_distance_list":isi_distance_list, 
                      "spike_distance_list":spike_distance_list}
        
        with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
            pickle.dump(test_stats, f)
            f.close()
        with open(os.path.join(savepath,'saved_stats.pkl'), 'wb') as f:
            pickle.dump(all_stats, f)
            f.close()