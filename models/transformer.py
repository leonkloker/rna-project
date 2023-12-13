import torch
import torch.nn as nn

from models.utils import *

class Transformer(nn.Module):
    def __init__(self, d_features, d_model, n_head, n_encoder_layers, n_decoder_layers, 
                 d_feedforward, dropout=0.0, activation='gelu'):
        super(Transformer, self).__init__()

        self.embedding = nn.Linear(d_features, d_model)
        self.positional_encoding = PositionalEncodingNLP(d_model, dropout=dropout, max_len=500)

        if n_encoder_layers == 0:
            self.transformer = nn.Transformer(d_model, n_head, n_encoder_layers, n_decoder_layers,
                                           d_feedforward, dropout, custom_encoder=MyIdentityEncoder(),
                                           batch_first=True, activation=activation)
        elif n_decoder_layers == 0:
            self.transformer = nn.Transformer(d_model, n_head, n_encoder_layers, n_decoder_layers,
                                           d_feedforward, dropout, custom_decoder=MyIdentityDecoder(),
                                           batch_first=True, activation=activation)
        else:
            self.transformer = nn.Transformer(d_model, n_head, n_encoder_layers, n_decoder_layers,
                                           d_feedforward, dropout, batch_first=True,
                                           activation=activation)
            
        self.fc = nn.Sequential(nn.Linear(d_model, 64),
                                      nn.Dropout(dropout),
                                      nn.GELU(),
                                      nn.Linear(64, 16),
                                      nn.Dropout(dropout),
                                      nn.GELU(),
                                      nn.Linear(16, 1),
                                      nn.Sigmoid())
        
    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        output = self.transformer(src, tgt, tgt_mask=None, src_mask=None, memory_mask=None)
        output = self.fc(output)

        return output

class MyIdentityEncoder(nn.Module):
    def __init__(self):
        super(MyIdentityEncoder, self).__init__()

    def forward(self, x, **kwargs):
        return x

class MyIdentityDecoder(nn.Module):
    def __init__(self):
        super(MyIdentityDecoder, self).__init__()

    def forward(self, x, y, **kwargs):
        return x
