import math
import numpy as np
import random
import re
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader, Subset

def generate_mask(sz1, sz2=None, window=-1):
        # square mask
        if sz2 is None:
            sz2 = sz1
        
        # no mask
        if window == -2:
            mask = None

        # mask when all past history is available
        elif window == -1:
            mask = (torch.tril(torch.ones(sz1, sz2)) == 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz1, sz2)
            for i in range(sz1):
                mask[i, max(0, i - window + 1) : min(i + 1, sz2)] = 1
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

        return mask

def generate_mask_bool(sz1, sz2=None, window=-1):
        # square mask
        if sz2 is None:
            sz2 = sz1
        
        # no mask
        if window == -2:
            mask = None

        # mask when all past history is available
        elif window == -1:
            mask = torch.logical_not((torch.tril(torch.ones(sz1, sz2)) == 1).bool())
        
        # mask when only a window of past history is available
        else:
            mask = torch.zeros(sz1, sz2)
            for i in range(sz1):
                mask[i, max(0, i - window + 1) : min(i + 1, sz2)] = 1
            mask = torch.logical_not(mask.bool())

        return mask

class PositionalEncodingNLP(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncodingNLP, self).__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLinear(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super(PositionalEncodingLinear, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model)
        pos_encoding[:,:] = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) / max_len
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingSinusoidal(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super(PositionalEncodingSinusoidal, self).__init__()
        self.dropout = nn.Dropout(dropout)

        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float)
        pos_encoding[:,:] = -torch.cos(torch.pi * (position / max_len)).unsqueeze(1)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        x = x + self.pos_encoding[:x.size(-2), :]
        return self.dropout(x)
    
class PositionalEncodingLearned(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.0):
        super(PositionalEncodingLearned, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.linear_pos_encoding = PositionalEncodingLinear(d_model, max_len, dropout)
        self.linear = nn.Sequential(nn.Linear(d_model, d_model), 
                                    nn.ReLU(), 
                                    nn.Linear(d_model, d_model),
                                    nn.Tanh())

    def forward(self, x):
        x = x + self.linear(self.linear_pos_encoding.pos_encoding)[:x.size(-2), :]
        return self.dropout(x)

class CustomMAEloss(nn.Module):
    def __init__(self):
        super(CustomMAEloss, self).__init__()

    def forward(self, prediction, ground_truth):
        loss = torch.nanmean(torch.abs(prediction - ground_truth))
        return loss

class CustomBCEloss(nn.Module):
    def __init__(self):
        super(CustomBCEloss, self).__init__()

    def forward(self, inputs, targets):
        mask = ~torch.isnan(targets) & ~torch.isnan(inputs)
        filtered_inputs = inputs[mask]
        filtered_targets = targets[mask]
        loss = F.binary_cross_entropy_with_logits(filtered_inputs, filtered_targets)
        return loss
    
def pearsonCorrelation(prediction, ground_truth):
    if ground_truth.dim() == 1:
        nan_mask = ~torch.isnan(ground_truth)
        pearson_avg = torch.corrcoef(torch.stack([prediction[nan_mask], ground_truth[nan_mask]]))[0, 1]
        pearson_med = pearson_avg
    else:
        pearson_coeffs = torch.zeros(prediction.shape[0])
        for i in range(prediction.shape[0]):
            nan_mask = ~torch.isnan(ground_truth[i])
            pearson_coeffs[i] = torch.corrcoef(torch.stack([prediction[i][nan_mask], ground_truth[i][nan_mask]]))[0, 1]
        pearson_avg = torch.mean(pearson_coeffs)
        pearson_med = torch.median(pearson_coeffs)

    return pearson_avg, pearson_med

def get_random_subset_loader(dataset, batch_size, subset_fraction=0.1):
    n = len(dataset)
    subset_indices = random.sample(range(n), int(n * subset_fraction))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader
