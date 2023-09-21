import torch
from torch import nn
from .sparsemax import *
class att_encoder(nn.Module):
    def __init__(self,
                 rnn,
                 num_rnn_inputs,
                 num_rnn_layers,
                 num_rnn_hidden,
                 bidirectional = False,
                 act='leakyrelu',
                 attention=False,
                 fullattention=False,
                 embedding_size=5,
                 window_size=50,
                 context_dense_size=10,
                 num_stimuli_condition = 23,
                 sparse=False,
                 **params
                ):
        super(att_encoder, self).__init__()

        rnn_units = {'lstm':nn.LSTM, 'gru':nn.GRU, 'rnn':nn.RNN}
        self.rnn_hidden = num_rnn_hidden
        #activations = {'relu': nn.ReLU, 'sigmoid': nn.Sigmoid, 'tanh': nn.Tanh, 'leakyrelu':nn.LeakyReLU}
        #tot_inputs = num_rnn_inputs
        
        rnn_unit = rnn_units[rnn]
        self.rnn_net = rnn_unit(input_size=num_rnn_inputs, 
                                hidden_size=num_rnn_hidden, 
                                num_layers=num_rnn_layers,
                                batch_first = True,
                                bidirectional = bidirectional
                               )
        
        self.attention = attention
        self.fullattention = fullattention
        if self.attention:
            self.rnn_net_att = rnn_unit(input_size=num_rnn_inputs, 
                                hidden_size=num_rnn_hidden, 
                                num_layers=num_rnn_layers,
                                batch_first = True,
                                bidirectional = bidirectional
                               )
        
            self.spatial_embed = nn.Linear(window_size,embedding_size)
            self.spatial_dense = nn.Linear(embedding_size+num_rnn_hidden+num_stimuli_condition,1)
            self.spatial_act = nn.Tanh()
            self.spatial_softmax = Sparsemax(dim=1) if sparse else nn.Softmax(dim=1) 
            
            self.temporal_dense = nn.Linear(num_rnn_hidden, 1)
            self.temporal_act = nn.Tanh()
            self.temporal_softmax = nn.Softmax(dim=1)
            
            self.final_dense = nn.Linear(num_rnn_hidden, context_dense_size)
            self.final_act = nn.ReLU()
            
        elif self.fullattention:
            self.rnn_net_att = rnn_unit(input_size=num_rnn_inputs, 
                                hidden_size=num_rnn_hidden, 
                                num_layers=num_rnn_layers,
                                batch_first = True,
                                bidirectional = bidirectional
                               )
        
            self.spatial_embed = nn.Linear(window_size,embedding_size)
            self.spatial_dense = nn.Linear(embedding_size+num_rnn_hidden+num_stimuli_condition,1)
            self.spatial_act = nn.Softplus()
            self.spatial_softmax = nn.Softmax(dim=1)
            
            self.temporal_dense = nn.Linear(num_rnn_hidden, 1)
            self.temporal_act = nn.Tanh()
            self.temporal_softmax = nn.Softmax(dim=1)
            
            self.final_dense = nn.Linear(num_rnn_hidden, context_dense_size)
            self.final_act = nn.ReLU()
        
    def forward(self, x, stimuli=None, hidden=None, get_temporal=False):
        rnn_out, hidden = self.rnn_net(x, hidden)
        rnn_out_constig = rnn_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
        
        if self.attention:
            # spatial attention
            spatial_embedding = self.get_spatial_embedding(x)
            betai = self.get_spatial_beta(x,rnn_out_constig, stimuli, spatial_embedding=spatial_embedding)
            
            # temporal attention
            alphai = self.get_temporal_alpha(rnn_out)
            
            spatial_context = torch.transpose((betai/betai.mean(1).unsqueeze(1))*torch.transpose(x, 1,2), 1,2)
            spatial_temporal_context = (alphai/alphai.mean(1).unsqueeze(1))*spatial_context
            spatial_temporal_out, spatial_temporal_hidden = self.rnn_net_att(spatial_temporal_context, None)
            spatial_temporal_embedding = spatial_temporal_out[:,-1,:].contiguous().view(-1, self.rnn_hidden)
            spatial_temporal_embedding = self.final_dense(spatial_temporal_embedding)
            att_weights = betai
            if get_temporal:
                att_weights = alphai
            return spatial_temporal_embedding, hidden, att_weights
        
        return rnn_out_constig, hidden, None
    
    def get_spatial_embedding(self, x):
        x_transpose = torch.transpose(x, 1,2)
        spatial_embedding = self.spatial_embed(x_transpose)
        
        return spatial_embedding
    
    def get_spatial_beta(self, x, rnn_out_constig, stimuli, spatial_embedding=None):
        if self.attention:
            spatial_embedding = spatial_embedding if spatial_embedding is not None else self.get_spatial_embedding(x)
            spatial_concat = torch.concat([spatial_embedding, 
                                           rnn_out_constig.unsqueeze(1).repeat(1,spatial_embedding.shape[1],1), 
                                           stimuli.unsqueeze(1).repeat(1,spatial_embedding.shape[1],1)], -1)
            ei = self.spatial_act(self.spatial_dense(spatial_concat))
            betai = self.spatial_softmax(ei)
            return betai
        else:
            raise NotImplementedError("Not attention based RNN encoder")
        
    def get_temporal_alpha(self, rnn_out):
        if self.attention:
            temporal_embed = self.temporal_dense(rnn_out)
            ai = self.temporal_act(temporal_embed)
            alphai = self.temporal_softmax(ai)
            return alphai
        else:
            raise NotImplementedError("Not attention based RNN encoder")
        
        
