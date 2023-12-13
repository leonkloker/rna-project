import torch
import torch.nn as nn

import models.cnn

class GRU(nn.Module):
    def __init__(self, d_model, n_layers, d_features=640, conv_dims=[64, 16, 1], dropout=0.0,
                 n_heads=1, embedding=False):
        super(GRU, self).__init__()

        self.gru = nn.GRU(input_size=d_model, hidden_size=d_model, 
                          num_layers=n_layers, bidirectional=True,
                          batch_first=True, dropout=dropout)
        
        self.n_heads = n_heads

        if embedding:
            self.embedding = nn.Embedding(25, d_model, padding_idx=1)

        conv_dims = [2*d_model] + conv_dims
        self.cnn = nn.ModuleList()
        for i in range(self.n_heads):
            self.cnn.append(models.cnn.Conv1DModel(conv_dims, dropout=dropout))
                                 
    def forward(self, x):
        if hasattr(self, 'embedding'):
            x = self.embedding(x.int())

        x, _ = self.gru(x)
        res = []
        for i in range(self.n_heads):
            res.append(self.cnn[i](x).squeeze(-1))
        return res
    
class GRU_FC(nn.Module):
    def __init__(self, d_features, d_model, n_layers, dropout=0.0):
        super(GRU_FC, self).__init__()

        self.gru = nn.GRU(input_size=d_features, hidden_size=d_model, 
                          num_layers=n_layers, bidirectional=True,
                          batch_first=True, dropout=dropout)
        
        self.fc = nn.Sequential(
            nn.Linear(2*d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
                                 
    def forward(self, x):
        x, _ = self.gru(x)
        return self.fc(x)