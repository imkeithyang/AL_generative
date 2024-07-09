# AL_generative

Code Repository for our paper: **Neuron synchronization analyzed through spatial-temporal attention** ([bioRxiv]()). This repo contains the implementation of conditional normalizing flow with a spatial attention module that learns semi-interpretable weights to represent neural synchronization in *Manduca sexta*. 

## Acknowledgment
The normalizing flow implementation is adapted from [https://github.com/ikostrikov/pytorch-flows](https://github.com/ikostrikov/pytorch-flows).           
There is also a sparsemax implementation adapted from [https://github.com/aced125/sparsemax](https://github.com/aced125/sparsemax).         
We thank Dr. Lei for the code that classifies LNs and PNs. Our adaptation to the original matlab code is in ```PoissonSurprise/```         

Use the following command to train ```python main_attflow_cond_stim.py -f yamlfilepath -d cuda:index```

Use bash files to train a specific moth's models through config/moth_id/: ```bash train_attflow_cond_stim.sh``` and input config directory and gpu device to start at (input 0 for default).
