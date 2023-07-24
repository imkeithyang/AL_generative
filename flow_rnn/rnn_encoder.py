import torch
from torch import nn

class rnn_encoder(nn.Module):
    def __init__(self,
                 rnn,
                 num_rnn_inputs,
                 num_rnn_layers,
                 num_rnn_hidden,
                 bidirectional = False):
        super(rnn_encoder, self).__init__()

        rnn_units = {'lstm':nn.LSTM, 'gru':nn.GRU, 'rnn':nn.RNN}
        self.rnn_hidden = num_rnn_hidden
        
        rnn_unit = rnn_units[rnn]
        self.rnn_net = rnn_unit(input_size=num_rnn_inputs, 
                                hidden_size=num_rnn_hidden, 
                                num_layers=num_rnn_layers,
                                batch_first = True,
                                bidirectional = bidirectional
                               )
        
    def forward(self, x, hidden=None):
        rnn_out, hidden = self.rnn_net(x, hidden)
        rnn_out_constig = rnn_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        
        return rnn_out_constig, hidden
    
    
        
        
