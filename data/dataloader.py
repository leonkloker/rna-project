import numpy as np
import os
import sys
from torch.utils.data import Dataset
import torch

sys.path.insert(0, '../')

from models.utils import *

TRAIN_SPLIT = 0.9
VAL_SPLIT = 0.05
TEST_SPLIT = 0.05

class DatasetRNA(Dataset):
    def __init__(self, root_dir, embedding=True, mode='train', N=0, secondary=False):
        self.x_dir = root_dir
        self.secondary = secondary

        if embedding:
            self.y1_dir = root_dir.replace("fm_embeddings", "2A3_MaP")
            self.y2_dir = root_dir.replace("fm_embeddings", "DMS_MaP")
            self.secondary_dir = root_dir.replace("fm_embeddings", "secondary")
        else:
            self.y1_dir = root_dir.replace("sequences", "2A3_MaP")
            self.y2_dir = root_dir.replace("sequences", "DMS_MaP")
            self.secondary_dir = root_dir.replace("sequences", "secondary")

        self.x_list = os.listdir(self.x_dir)
        self.y1_list = os.listdir(self.y1_dir)
        self.y2_list = os.listdir(self.y2_dir)
        self.mode = mode
        
        if mode == 'train':
            if N == 0:
                self.length = int(len(self.x_list) * TRAIN_SPLIT)
            else:
                self.length = int(N * TRAIN_SPLIT)
        
        elif mode == 'val':
            if N == 0:
                self.length = int(len(self.x_list) * VAL_SPLIT)
            else:
                self.length = int(N * VAL_SPLIT)

        elif mode == 'test':
            if N == 0:
                self.length = int(len(self.x_list) * TEST_SPLIT)
            else:
                self.length = int(N * TEST_SPLIT)

    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        if self.mode == 'val':
            idx = idx + int((self.length / VAL_SPLIT) * TRAIN_SPLIT)
        if self.mode == 'test':
            idx = idx + int((self.length / TEST_SPLIT) * (TRAIN_SPLIT + VAL_SPLIT))

        x = np.load(os.path.join(self.x_dir, self.x_list[idx]))
        y1 = np.load(os.path.join(self.y1_dir, self.y1_list[idx]))
        y2 = np.load(os.path.join(self.y2_dir, self.y2_list[idx]))            

        x = x["x"]
        y1 = y1["x"][:x.shape[0]]
        y2 = y2["x"][:x.shape[0]]

        if self.secondary:
            file = os.path.join(self.secondary_dir, "{}.npz".format(idx))
            if os.path.exists(file):
                s = np.load(file)
                s = s["x"][:x.shape[0]]
            else:
                s = np.full(y1.shape, np.nan)
        else:
            s = np.full(y1.shape, np.nan)

        return x, y1, y2, s
    