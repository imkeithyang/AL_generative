import yaml
import torch
import pickle
import copy

from utils import *
from deepAR import *

args = get_parser().parse_args()
if args.device is None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else: 
    device = torch.device(args.device)
if args.trainflow in set([0,"0", "False", "F"]):
    trainflow = 0
else: 
    trainflow = 1
    
try:
    yaml_filepath = args.filename
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
except:
    yaml_filepath = "AL_generative/config/deepAR.yaml"
    with open(yaml_filepath, 'r') as f:
        cfg = yaml.load(f, yaml.SafeLoader)
        
try:
    n_runs = cfg['n_runs']
except KeyError:
    n_runs = 5
try:
    n_tries = cfg['n_tries']
except KeyError:
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
    
    # check if we have done previous preprocessing or not
    for stimuli in range(0,23):
        important_index = None
        savepath, plot_savepath, net_savepath,exp = format_directory(cfg_temp, None, stimuli)
        important_index_file = os.path.join("/hpc/home/pc266/AL_generative",exp,'important_index_deepAR.pkl')
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
            print("Found Existing Index For Target {}, Stimuli {}: {} ".format(target, stimuli, important_index))
        else:
            print("Important Index For Target {}, Stimuli {} Not Found...Preprocessing".format(target, stimuli))
    
        if important_index is None:
            store_loss = []
            for run in range(n_runs):
                initialized, _, _ = setup_deepAR(cfg_temp, device, run, stimuli_index=stimuli)
                loss_dict, loss = train_multiple_deepAR(**initialized)
                print(loss)
                store_loss.append(loss)
                
            loss_mean = np.array(store_loss).sum(0)/n_runs
            loss_mean_target = loss_mean[target]
            loss_ratio = loss_mean_target/loss_mean
            important_index = np.where(loss_ratio > important_threshold)[0]
            important_dict = {"loss": store_loss, "loss_ratio": loss_ratio}
            if not os.path.exists(exp):
                os.makedirs(exp)
            with open(os.path.join(exp,'important_index_deepAR.pkl'), 'wb') as f:
                pickle.dump(important_dict, f)
                f.close()
            print("Target = {} | Important Index = {}".format(target,important_index))