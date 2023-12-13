import datetime
import fm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
torch.cuda.manual_seed(42)

from models.gru import *
from data.dataloader import *
from models.utils import *

# Model parameters
D_FEATURES = 640
D_MODEL = 192
N_LAYERS = 3

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
LEARNING_RATE_FT = 5e-5
LAYERS_FT = [10, 11]
BATCH_SIZE = 96

# Load foundation model
foundation_model, alphabet = fm.pretrained.rna_fm_t12()
def fm_forward(x):
    x = foundation_model.embed_tokens(x)
    for layer in foundation_model.layers:
        x, _ = layer(x)
    return x

# Log name
date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
name = 'finetune{}_gru_dim{}_layers{}_epochs{}_lr{:.0E}_{}'.format(
            "_".join(str(i) for i in LAYERS_FT), D_FEATURES, N_LAYERS, NUM_EPOCHS, LEARNING_RATE, date)

# Initialize log file and tensorboard writer
file = open("../outputs/logs/{}.txt".format(name), "w")
writer = SummaryWriter(log_dir="../outputs/tensorboards/{}".format(name))

# Initialize data loaders
data_dir = '../data/sequences/'
dataloader_train = DataLoader(DatasetRNA(data_dir, mode='train', embedding=False), batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(DatasetRNA(data_dir, mode='val', embedding=False), batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(DatasetRNA(data_dir, mode='test', embedding=False), batch_size=BATCH_SIZE, shuffle=True)

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using cuda", file=file)
else:
    device = torch.device('cpu')
    print("Using CPU", file=file)

# Finetune layers
params = []
for string, param in foundation_model.named_parameters():
    if any([str(layer) in string for layer in LAYERS_FT]):
        param.requires_grad = True
        params.append(param)
    else:
        param.requires_grad = False

foundation_model = foundation_model.to(device)
print(foundation_model, file=file)

# Initialize model
gru = GRU(D_FEATURES, D_MODEL, N_LAYERS)
gru = gru.to(device)
model_info = summary(gru, input_size=(BATCH_SIZE, 200, 640))
print(model_info, file=file)
file.flush()

# Initialize optimizer, scheduler, and loss function
optimizer = optim.AdamW([{'params': params, 'lr': LEARNING_RATE_FT},
                          {'params': gru.parameters(), 'lr': LEARNING_RATE}])
scheduler = OneCycleLR(optimizer, max_lr=[10*LEARNING_RATE_FT, 10*LEARNING_RATE], epochs=NUM_EPOCHS, steps_per_epoch=len(dataloader_train))
cost = CustomMAEloss()
val_loss_min = np.Inf

# Training loop
for epoch in range(NUM_EPOCHS):
    for x, y1, y2 in dataloader_train:
        x = x.int()
        x = x.to(device)
        y1 = y1.to(device)

        optimizer.zero_grad()

        x = fm_forward(x)  
        yhat = gru(x).squeeze(-1)
        loss = cost(yhat, y1)

        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(gru.parameters(), 5)
        nn.utils.clip_grad_norm_(params, 5)
        optimizer.step()

    # Validation
    foundation_model.eval()
    gru.eval()
    val_loss = 0
    pearson_avgs = []
    pearson_medians = []

    with torch.no_grad():
        for x, y1, y2 in dataloader_val:
            x = x.int()
            x = x.to(device)
            y1 = y1.to(device)

            x = fm_forward(x)
            yhat = gru(x).squeeze(-1)
            val_loss += cost(yhat, y1)

            pearson_avg, pearson_median = pearsonCorrelation(yhat, y1)
            pearson_avgs.append(pearson_avg)
            pearson_medians.append(pearson_median)

    val_loss = val_loss / len(dataloader_val)
    pearson_avg = torch.mean(torch.stack(pearson_avgs))
    pearson_median = torch.median(torch.stack(pearson_medians))

    # Log validation loss
    writer.add_scalar("MAE/train", loss, epoch)
    writer.add_scalar("MAE/val", val_loss, epoch)
    writer.add_scalar("Pearson/val_avg", pearson_avg, epoch)
    writer.add_scalar("Pearson/val_median", pearson_median, epoch)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, learning rate: {optimizer.param_groups[1]['lr']}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, train MAE: {loss.item()}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val MAE: {val_loss}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val Pearson: {pearson_avg}", file=file)
    file.flush()

    # Save model if validation loss is lower than previous minimum
    if val_loss < val_loss_min:
        torch.save(gru.state_dict(), 
                   '../outputs/models/{}_head.pth'.format(name))
        torch.save(foundation_model.state_dict(),
                     '../outputs/models/{}_fm.pth'.format(name))
        val_loss_min = val_loss

    # Update learning rate
    foundation_model.train()
    gru.train()
    scheduler.step()
    writer.add_scalar("learning_rate", optimizer.param_groups[1]['lr'], epoch)

print("Training finished", file=file)
print("\nEvaluating best model on test set", file=file)
file.flush()

# Load best model and evaluate on test set
foundation_model.load_state_dict(torch.load('../outputs/models/{}_fm.pth'.format(name)))
gru.load_state_dict(torch.load('../outputs/models/{}_head.pth'.format(name)))
foundation_model.eval()
gru.eval()

test_loss = 0
pearson_avgs = []
pearson_medians = []
with torch.no_grad():
    for x, y1, y2 in dataloader_test:
        x = x.int()
        x = x.to(device)
        y1 = y1.to(device)

        x = fm_forward(x)
        yhat = gru(x).squeeze(-1)
        test_loss += cost(yhat, y1)

        pearson_avg, pearson_median = pearsonCorrelation(yhat, y1)
        pearson_avgs.append(pearson_avg)
        pearson_medians.append(pearson_median)

test_loss = test_loss / len(dataloader_test)
pearson_avg = torch.mean(torch.stack(pearson_avgs))
pearson_median = torch.median(torch.stack(pearson_medians))

# Log test loss
writer.add_scalar("MAE/test", test_loss, epoch)
writer.add_scalar("Pearson/test_avg", pearson_avg, epoch)
writer.add_scalar("Pearson/test_median", pearson_median, epoch)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test MAE: {val_loss}", file=file)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test average Pearson: {pearson_avg}", file=file)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test median Pearson: {pearson_median}", file=file)
file.flush()

