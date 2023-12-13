import datetime
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gru import *
from data.dataloader import *
from models.utils import *

# Model parameters
D_FEATURES = 640
D_MODEL = 128
N_LAYERS = 3
N_HEADS = 1

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

SAMPLE = 3

# Log name
date = "2023-12-06-17:26"
name = 'gru_dim{}_layers{}_epochs{}_lr{:.0E}_{}'.format(
            D_FEATURES, N_LAYERS, NUM_EPOCHS, LEARNING_RATE, date)

# Initialize data loaders
data_dir = '../data/fm_embeddings/'
dataloader_train = DataLoader(DatasetRNA(data_dir, mode='train', secondary=False), batch_size=1, shuffle=True)

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# Initialize model
model = GRU_FC(D_FEATURES, D_MODEL, N_LAYERS)
model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
model = model.to(device)

for i, (x, y1, y2) in enumerate(dataloader_train):
    x = x.to(device)
    yhat = model(x).cpu().squeeze(-1).squeeze(0).detach()
    y1 = y1.squeeze(0).detach()
    break

y1 = y1[1:]
yhat = yhat[:-1]
mask = ~torch.isnan(y1)
plt.figure()
plt.plot(np.arange(y1.shape[0])[mask], y1[mask], label='Target')
plt.plot(np.arange(yhat.shape[0])[mask], yhat[mask], label='Prediction')
plt.legend(loc='upper right')
plt.xlabel('Nucleotide position')
plt.ylabel('SHAPE')
plt.title('MAE: {:.4f}, Pearson {:.4f}'.format(CustomMAEloss()(yhat, y1).item(), pearsonCorrelation(yhat, y1)[0])) 
plt.tight_layout()
plt.savefig('../outputs/figures/prediction.png'.format(SAMPLE), dpi=300)
