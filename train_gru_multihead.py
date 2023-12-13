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

from models.gru import *
from data.dataloader import *
from models.utils import *

# Model parameters
D_FEATURES = 640
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 3

# Training parameters
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

# Log name
date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M")
name = 'gru_cnn_heads{}_dim{}_layers{}_epochs{}_lr{:.0E}_{}'.format(
            N_HEADS, D_FEATURES, N_LAYERS, NUM_EPOCHS, LEARNING_RATE, date)

# Initialize log file and tensorboard writer
file = open("../outputs/logs/{}.txt".format(name), "w")
writer = SummaryWriter(log_dir="../outputs/tensorboards/{}".format(name))

# Initialize data loaders
data_dir = '../data/fm_embeddings/'
dataloader_train = DataLoader(DatasetRNA(data_dir, mode='train', secondary=True), batch_size=BATCH_SIZE, shuffle=True)
dataloader_val = DataLoader(DatasetRNA(data_dir, mode='val', secondary=True), batch_size=BATCH_SIZE, shuffle=True)
dataloader_test = DataLoader(DatasetRNA(data_dir, mode='test', secondary=True), batch_size=BATCH_SIZE, shuffle=True)

# Check if cuda is available
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Using cuda", file=file)
else:
    device = torch.device('cpu')
    print("Using CPU", file=file)

# Initialize model
model = GRU(D_FEATURES, D_MODEL, N_LAYERS, n_heads=N_HEADS)
model = model.to(device)
model_info = summary(model, input_size=(BATCH_SIZE, 200, D_FEATURES))
print(model_info, file=file)
file.flush()

# Initialize optimizer, scheduler, and loss function
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=5, min_lr=1e-7)
cost1 = CustomMAEloss()
cost2 = CustomBCEloss()
val_loss_min = np.Inf

# Training loop
for epoch in range(NUM_EPOCHS):
    for x, y1, y2, s in dataloader_train:
        x = x.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        s = s.to(device)

        optimizer.zero_grad()
        yhat = model(x)
        loss = cost1(yhat[0], y1)
        loss += cost1(yhat[1], y2)
        loss += cost2(yhat[2], s)

        # Backpropagation
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0
    val_loss1 = 0
    val_loss2 = 0
    val_losss = 0
    pearson_avgs1 = []
    pearson_medians1 = []
    pearson_avgs2 = []
    pearson_medians2 = []

    with torch.no_grad():
        for x, y1, y2, s in dataloader_val:
            x = x.to(device)
            y1 = y1.to(device)
            y2 = y2.to(device)
            s = s.to(device)

            yhat = model(x)
            val_loss1 += cost1(yhat[0], y1)
            val_loss2 += cost1(yhat[1], y2)
            val_losss += cost2(yhat[2], s)
            val_loss += val_loss1 + val_loss2 + val_losss

            pearson_avg, pearson_median = pearsonCorrelation(yhat[0], y1)
            pearson_avgs1.append(pearson_avg)
            pearson_medians1.append(pearson_median)

            pearson_avg, pearson_median = pearsonCorrelation(yhat[1], y2)
            pearson_avgs2.append(pearson_avg)
            pearson_medians2.append(pearson_median)

    val_loss = val_loss / len(dataloader_val)
    val_loss1 = val_loss1 / len(dataloader_val)
    val_loss2 = val_loss2 / len(dataloader_val)
    val_losss = val_losss / len(dataloader_val)
    pearson_avg1 = torch.mean(torch.stack(pearson_avgs1))
    pearson_median1 = torch.median(torch.stack(pearson_medians1))
    pearson_avg2 = torch.mean(torch.stack(pearson_avgs2))
    pearson_median2 = torch.median(torch.stack(pearson_medians2))

    # Log validation loss
    writer.add_scalar("Loss/train", loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("MAE/val1", val_loss1, epoch)
    writer.add_scalar("MAE/val2", val_loss2, epoch)
    writer.add_scalar("BCE/val", val_losss, epoch)
    writer.add_scalar("Pearson/val_avg1", pearson_avg1, epoch)
    writer.add_scalar("Pearson/val_median1", pearson_median1, epoch)
    writer.add_scalar("Pearson/val_avg2", pearson_avg2, epoch)
    writer.add_scalar("Pearson/val_median2", pearson_median2, epoch)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, learning rate: {optimizer.param_groups[0]['lr']}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, train loss: {loss.item()}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val loss: {val_loss}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val MAE 1: {val_loss1}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val MAE 2: {val_loss2}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val BCE: {val_losss}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val Pearson 1: {pearson_median1}", file=file)
    print(f"Epoch {epoch+1} / {NUM_EPOCHS}, val Pearson 2: {pearson_median2}", file=file)
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
test_loss1 = 0
test_loss2 = 0
test_losss = 0
pearson_avgs1 = []
pearson_medians1 = []
pearson_avgs2 = []
pearson_medians2 = []
with torch.no_grad():
    for x, y1, y2, s in dataloader_test:
        x = x.to(device)
        y1 = y1.to(device)
        y2 = y2.to(device)
        s = s.to(device)

        yhat = model(x)
        test_loss1 += cost1(yhat[0], y1)
        test_loss2 += cost1(yhat[1], y2)
        test_losss += cost2(yhat[2], s)
        test_loss += test_loss1 + test_loss2 + test_losss

        pearson_avg, pearson_median = pearsonCorrelation(yhat[0], y1)
        pearson_avgs1.append(pearson_avg)
        pearson_medians1.append(pearson_median)
        pearson_avg, pearson_median = pearsonCorrelation(yhat[1], y2)
        pearson_avgs2.append(pearson_avg)
        pearson_medians2.append(pearson_median)

test_loss = test_loss / len(dataloader_test)
test_loss1 = test_loss1 / len(dataloader_test)
test_loss2 = test_loss2 / len(dataloader_test)
test_losss = test_losss / len(dataloader_test)
pearson_avg1 = torch.mean(torch.stack(pearson_avgs1))
pearson_median1 = torch.median(torch.stack(pearson_medians1))
pearson_avg2 = torch.mean(torch.stack(pearson_avgs2))
pearson_median2 = torch.median(torch.stack(pearson_medians2))

# Log test loss
writer.add_scalar("Loss/test", test_loss, epoch)
writer.add_scalar("MAE/test1", test_loss1, epoch)
writer.add_scalar("MAE/test2", test_loss2, epoch)
writer.add_scalar("BCE/test", test_losss, epoch)
writer.add_scalar("Pearson/test_avg1", pearson_avg1, epoch)
writer.add_scalar("Pearson/test_median1", pearson_median1, epoch)
writer.add_scalar("Pearson/test_avg2", pearson_avg2, epoch)
writer.add_scalar("Pearson/test_median2", pearson_median2, epoch)

print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test loss: {test_loss}", file=file)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test MAE 1: {test_loss1}", file=file)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test MAE 2: {test_loss2}", file=file)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test BCE: {test_losss}", file=file)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test Pearson 1: {pearson_median1}", file=file)
print(f"Epoch {epoch+1} / {NUM_EPOCHS}, test Pearson 2: {pearson_median2}", file=file)
file.flush()
