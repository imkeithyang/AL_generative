import torch
from torch import nn


class deepAR_net(nn.Module):
    def __init__(self,
                 rnn,
                 num_rnn_inputs,
                 num_rnn_layers,
                 num_rnn_hidden,
                 num_dense_layers,
                 num_dense_hidden,
                 num_outputs,
                 stimuli_conditional,
                 bidirectional = False,
                 act='leakyrelu',
                 **params
                ):
        super(deepAR_net, self).__init__()

        rnn_units = {'lstm':nn.LSTM, 'gru':nn.GRU, 'rnn':nn.RNN}
        self.rnn_hidden = num_rnn_hidden
        #activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leakyrelu':nn.LeakyReLU}
        #tot_inputs = num_rnn_inputs
        
        rnn_unit = rnn_units[rnn]
        self.rnn_net = rnn_unit(input_size=num_rnn_inputs, 
                                hidden_size=num_rnn_hidden, 
                                num_layers=1 + num_rnn_layers,
                                batch_first = True,
                                bidirectional = bidirectional
                               )
        
        mu_act = nn.Softplus
        mu_input = num_rnn_hidden + stimuli_conditional
        mu_modules = [nn.Linear(mu_input, num_dense_hidden), mu_act()]
            
        for _ in range(num_dense_layers):
            mu_modules += [nn.Linear(num_dense_hidden, num_dense_hidden), mu_act()]
            if _ == num_dense_layers-1:
                mu_modules += [nn.Linear(num_dense_hidden, num_outputs), mu_act()]
                

        if num_dense_layers == 0:
            mu_modules = [nn.Linear(num_rnn_hidden, num_outputs, mu_act())]
            
        self.dense_mu = nn.Sequential(*mu_modules)
    
    def forward(self, x, q=None, t=None, hidden=None):
        rnn_out, hidden = self.rnn_net(x, hidden)
        rnn_out = rnn_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        # given a specific neuron, and a specific stimuli
        dense_in = torch.cat([rnn_out, q], -1) if q is not None else rnn_out
        dense_out = self.dense_mu(dense_in)
        return dense_out, hidden
    
    