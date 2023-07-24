from utils import *
from vae_flow import *

import torch
import yaml
import pickle


if __name__ == "__main__":
    import shutil
    global savepath
    global plot_savepath
    global net_savepath
    global device
    
    args = get_parser().parse_args()

    if args.device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else: 
        device = torch.device(args.device)

    yaml_filepath = args.filename
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
        

    all_stats = {'config':cfg, 'runs':[]}

    try:
        n_runs = cfg['n_runs']
    except KeyError:
        n_runs = 5
    try:
        n_tries = cfg['n_tries']
    except KeyError:
        n_tries = 1

    for run in range(0,n_runs):
        savepath, plot_savepath, net_savepath, exp = format_directory(cfg, run)
        make_directory(exp, savepath, plot_savepath, net_savepath)
        initialized, test_loader = setup(cfg, device, seed=run)
        initialized["paths"] = (savepath, plot_savepath, net_savepath)
        initialized["device"] = device
        all_stats, best_epoch = train(**initialized)
        
        vae_best = initialized["vae"]
        vae_best.load_state_dict(torch.load(net_savepath + "/vae.pt"))
        vae_best.eval()
        
        lw_best = initialized["lw"]
        lw_best.load_state_dict(torch.load(net_savepath + "/lw.pt"))
        lw_best.eval()
        
        flow_net_best = initialized["flow_net"]
        flow_net_best.load_state_dict(torch.load(net_savepath + "/flow_net.pt"))
        flow_net_best.eval()
        
        test_stats = validation(vae_best, lw_best, flow_net_best, test_loader, device)
        
        window_size     = initialized["window_size"]
        n_neurons       = initialized["n_neurons"]
        data_smooth     = initialized["data_smooth"]
        data_spike      = initialized["data_spike"]
        filler          = initialized["filler"]
        q               = initialized["q"]
        time_resolution = initialized["time_resolution"]
        
        spike_train = generate_spike_train(vae_best, lw_best, flow_net_best, device, 
                                           window_size, 
                                           n_neurons, 
                                           torch.zeros(1,1)+q, 
                                           time_resolution, data_spike, data_smooth, filler=filler)
        
        plot_spike_compare(
            initialized["data_spike"],
            spike_train.detach().cpu().numpy(), 
            neurons=np.arange(0,n_neurons),
            plot_savepath=savepath, 
            epoch="test", 
            q=q)
        
        with open(os.path.join(savepath,'test_stats.pkl'), 'wb') as f:
            pickle.dump(test_stats, f)
            f.close()
        with open(os.path.join(savepath,'saved_stats.pkl'), 'wb') as f:
            pickle.dump(all_stats, f)
            f.close()