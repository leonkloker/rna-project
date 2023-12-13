import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import tqdm

from models.gru import *
from models.lstm import *
from models.cnn import *
from models.transformer import *
from data.dataloader import *
from models.utils import *

# Initialize data loaders
data_dir = '../data/fm_embeddings/'
dataloader_train = DataLoader(DatasetRNA(data_dir, mode='train', secondary=False, N=1000), batch_size=1, shuffle=False)
dataloader_test = DataLoader(DatasetRNA(data_dir, mode='test', secondary=False, N=20000), batch_size=1, shuffle=False)

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Initialize model
name1 = 'gru_dim640_layers3_epochs100_lr1E-03_2023-12-06-17:26'
model1 = GRU_FC(640, 128, 3)
model1.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name1)))

name2 = 'lstm_dim640_layers3_epochs100_lr1E-03_2023-12-06-19:32'
model2 = LSTM(640, 128, 3)
model2.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name2)))

name3 = 'cnn_dim640_512_256_128_64_1_epochs100_lr1E-03_2023-12-06-15:22'
model3 = Conv1DModel([640, 512, 256, 128, 64, 1])
model3.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name3)))

name4 = 'transformer_embed256_enclayers3_declayers0_heads8_foward256_epochs100_lr1E-03_2023-12-06-11:52'
model4 = Transformer(640, 256, 8, 3, 0, 256)
model4.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name4)))

modelz = [model1, model2, model3, model4]
names = ["GRU", "LSTM", "CNN", "Transformer"]
col0 = []
col1 = []
col2 = []

for i, (x, y1, y2, s) in tqdm.tqdm(enumerate(dataloader_train)):
    for i, model in enumerate(modelz):
        y1 = y1.squeeze(0)
        if i == 3:
            yhat = model(x, x).squeeze(-1).squeeze(0).detach()
        else:
            yhat = model(x).squeeze(-1).squeeze(0).detach()
        col0.append(names[i])
        col1.append("train")
        p = pearsonCorrelation(yhat[:-1], y1[1:])[0].item()
        col2.append(p)

for i, (x, y1, y2, s) in tqdm.tqdm(enumerate(dataloader_test)):
    for i, model in enumerate(modelz):
        y1 = y1.squeeze(0)
        if i == 3:
            yhat = model(x, x).squeeze(-1).squeeze(0).detach()
        else:
            yhat = model(x).squeeze(-1).squeeze(0).detach()
        col0.append(names[i])
        col1.append("test")
        p = pearsonCorrelation(yhat[:-1], y1[1:])[0].item()
        col2.append(p)

df = pd.DataFrame({'Model' : col0, 'Pearson' : col2, 'Data': col1})
sns.violinplot(data=df, x="Model", y="Pearson", hue="Data", split=True, cut=0.05, gap=.1)
plt.title('Pearson correlation distribution')
plt.tight_layout()
plt.savefig('../outputs/figures/pearson_distribution.png')
