import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

torch.manual_seed(42)
torch.cuda.manual_seed(42)

from models.transformer import *
from data.dataloader import *
from models.utils import *

# Model parameters
N_FEATURES = 640
N_EMBEDDING = 512
N_HEADS = 8
N_FORWARD = 512
N_ENC_LAYERS = 0
N_DEC_LAYERS = 3

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

# Log name
date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
name = 'transformer_embed{}_enclayers{}_declayers{}_heads{}_foward{}_epochs{}_lr{:.0E}_{}'.format(
            N_EMBEDDING, N_ENC_LAYERS, N_DEC_LAYERS, N_HEADS, N_FORWARD, NUM_EPOCHS, LEARNING_RATE, date)

# Initialize log file and tensorboard writer
file = open("../outputs/logs/{}.txt".format(name), "w")
writer = SummaryWriter(log_dir="../outputs/tensorboards/{}".format(name))

# Initialize data loaders
data_dir = '../data/fm_embeddings/'
dataloader_train = DataLoader(DatasetEmbeddings(data_dir, mode='train'), batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(DatasetEmbeddings(data_dir, mode='val'), batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(DatasetEmbeddings(data_dir, mode='test'), batch_size=BATCH_SIZE, shuffle=True)

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using cuda", file=file)
else:
    device = torch.device('cpu')
    print("Using CPU", file=file)

# Initialize model
model = Transformer(N_FEATURES, N_EMBEDDING, N_HEADS, N_ENC_LAYERS, N_DEC_LAYERS, N_FORWARD)
model = model.to(device)
model_info = summary(model, input_size=[(BATCH_SIZE, 200, N_FEATURES), (BATCH_SIZE, 200, N_FEATURES)])
print(model_info, file=file)
file.flush()

# Initialize optimizer, scheduler, and loss function
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=1e-7)
cost = CustomMAEloss()
val_loss_min = np.Inf

# Training loop
for epoch in range(NUM_EPOCHS):
    for x, y1, y2 in dataloader_train:
        x = x.to(device)
        y1 = y1.to(device)

        optimizer.zero_grad()
        yhat = model(x, x).squeeze(-1)
        loss = cost(yhat, y1)

        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    pearson_avgs = []
    pearson_medians = []

    with torch.no_grad():
        for x, y1, y2 in dataloader_val:
            x = x.to(device)
            y1 = y1.to(device)

            yhat = model(x, x).squeeze(-1)
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
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, learning rate: {optimizer.param_groups[0]['lr']}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, train MAE: {loss.item()}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val MAE: {val_loss}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val Pearson: {pearson_median}", file=file)
    file.flush()

    # Save model if validation loss is lower than previous minimum
    if val_loss < val_loss_min:
        torch.save(model.state_dict(), 
                   '../outputs/models/{}.pth'.format(name))
        val_loss_min = val_loss

    # Update learning rate
    model.train()
    scheduler.step(val_loss)
    writer.add_scalar("learning_rate", optimizer.param_groups[0]['lr'], epoch)

print("Training finished", file=file)
print("\nEvaluating best model on test set", file=file)
file.flush()

# Load best model and evaluate on test set
model.load_state_dict(torch.load('../outputs/models/{}.pth'.format(name)))
model.eval()
test_loss = 0
pearson_avgs = []
pearson_medians = []
with torch.no_grad():
    for x, y1, y2 in dataloader_test:
        x = x.to(device)
        y1 = y1.to(device)

        yhat = model(x, x).squeeze(-1)
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
