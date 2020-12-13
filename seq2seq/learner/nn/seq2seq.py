"""
@author: jpzxshi
"""
import torch

from .module import StructureNN

class S2S(StructureNN):
    '''Seq2seq model.
    Input: [batch size, len_in, dim_in]
    Output: [batch size, len_out, dim_out]
    '''
    def __init__(self, dim_in, len_in, dim_out, len_out, hidden_size=10, cell='LSTM'):
        super(S2S, self).__init__()
        self.dim_in = dim_in
        self.len_in = len_in
        self.dim_out = dim_out
        self.len_out = len_out
        self.hidden_size = hidden_size
        self.cell = cell
        self.encoder = self.__init_encoder()
        self.decoder = self.__init_decoder()
        self.att_weights = self.__init_att_weights()
        self.out = self.__init_out()
        
    def forward(self, x):
        to_squeeze = True if len(x.size()) == 2 else False
        if to_squeeze:
            x = x.view(1, self.len_in, self.dim_in)
        zeros = torch.zeros([1, x.size(0), self.hidden_size], dtype=x.dtype, device=x.device)
        init_state = (zeros, zeros) if self.cell == 'LSTM' else zeros
        x, _ = self.encoder(x, init_state)
        x = torch.softmax(self.att_weights, dim=1) @ x
        x, _ = self.decoder(x, init_state)
        x = self.out(x)
        return x.squeeze(0) if to_squeeze else x
        
    def __init_encoder(self):
        if self.cell == 'RNN':
            return torch.nn.RNN(self.dim_in, self.hidden_size, batch_first=True)
        elif self.cell == 'LSTM':
            return torch.nn.LSTM(self.dim_in, self.hidden_size, batch_first=True)
        elif self.cell == 'GRU':
            return torch.nn.GRU(self.dim_in, self.hidden_size, batch_first=True)
        else:
            raise NotImplementedError
    
    def __init_decoder(self):
        if self.cell == 'RNN':
            return torch.nn.RNN(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.cell == 'LSTM':
            return torch.nn.LSTM(self.hidden_size, self.hidden_size, batch_first=True)
        elif self.cell == 'GRU':
            return torch.nn.GRU(self.hidden_size, self.hidden_size, batch_first=True)
        else:
            raise NotImplementedError
            
    def __init_att_weights(self):
        return torch.nn.Parameter(torch.zeros([self.len_out, self.len_in]))
    
    def __init_out(self):
        return torch.nn.Linear(self.hidden_size, self.dim_out)