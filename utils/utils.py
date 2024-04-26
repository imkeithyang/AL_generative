import os

def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-p",
        "--path",
        dest="path",
        default=None,
        help="dataset path",
        required=False,
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        default="/hpc/home/hy190/AL_generative/config/12142022/sparse-attflow/sparse-attflow-12.yaml",
        help="experiment definition file",
        metavar="FILE",
        required=False,
    )
    parser.add_argument(
        "-d",
        "--device",
        dest="device",
        default="cuda:0",
        help="experiment specified device",
        required=False,
    )
    parser.add_argument(
        "-seed",
        "--seed",
        dest="seed",
        default=42,
        help = "random seed",
        required = False,
    )
    parser.add_argument(
        "-shuffle",
        "--shuffle",
        dest="shuffle",
        default=False,
        help = "whether shuffle",
        required = False,
    )
    parser.add_argument(
        "-trainflow",
        "--trainflow",
        dest="trainflow",
        default="1",
        help="train flow",
        required=False,
    )
    return parser


def format_directory(cfg, run, stimuli=None, neuron_type=None, neuron=None):
    exp = cfg["data"]["path"].split("/")[-1]

    if cfg["shuffle"] == True:
        exp = "result_shuffle/" + exp[0:-4]  + (neuron_type if neuron_type else "")
    else:
        exp = "result/" + exp[0:-4]  + (neuron_type if neuron_type else "")
    
    if "use_component" in cfg["data"] and cfg["data"]["use_component"]:
        exp += "_use_comp"
    if "pre_stimuli" in cfg["data"]["path"]:
        exp += "_pre_stimuli"
        
    if "deepAR" in cfg:
        exp += "/deepAR/deepAR-{}/s_{}".format(neuron, stimuli)
        return None, None, None, exp
    
    if "rnn_encoder" in cfg:
        exp += "/rnnflow/rnnflow-{}".format(neuron)
    elif "att_encoder" in cfg:
        if "attention" in cfg["att_encoder"]["net"] and cfg["att_encoder"]["net"]["attention"]:
            if "sparse" in cfg["att_encoder"]["net"] and cfg["att_encoder"]["net"]["sparse"]:
                exp += "/sparse-attflow/sparse-attflow-{}".format(neuron)
            else:
                exp += "/attflow/attflow-{}".format(neuron)
        elif "fullattention" in cfg["att_encoder"]["net"] and cfg["att_encoder"]["net"]["fullattention"]:
            exp += "/attflow/attflow-{}".format(neuron)
    else:
        exp += "/vaeflow"
    
    if stimuli is None:
        exp += "-cond_stim"
    
    savepath = os.path.join(exp, "run_{}".format(run))
    plot_savepath = os.path.join(savepath, "by_stimuli/s_{}/plot".format(stimuli)) if stimuli is not None \
        else os.path.join(savepath, "plot")
    net_savepath = os.path.join("/scratch/pc266/AL_generative/", 
                                savepath, "by_stimuli/s_{}/net".format(stimuli)) if stimuli is not None else \
                   os.path.join("/scratch/pc266/AL_generative/", 
                                savepath, "net")
                                    
    return savepath, plot_savepath, net_savepath, exp

def make_directory(exp, savepath, plot_savepath, net_savepath):
    if not os.path.exists(exp):
        os.makedirs(exp)
    # skipping make path if path is None
    if savepath and not os.path.exists(savepath):
        os.makedirs(savepath)
    if plot_savepath and not os.path.exists(plot_savepath):
        os.makedirs(plot_savepath)
    if net_savepath and not os.path.exists(net_savepath):
        os.makedirs(net_savepath)
        
        

        
