import fm
import numpy as np
import os
import pandas as pd
import torch
import tqdm

# Set device
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Load model
print("Loading model...")
model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.to(device)
model.eval()

# Load RNA data to be transformed
print("Loading data...")
df = pd.read_csv("../../../csv/train_data.csv")

START = 0
END = 128 * 10**3

# Extract RNA sequences and labels
print("Extracting RNA sequences and labels...")
df1 = df[df["experiment_type"] == "2A3_MaP"]
df2 = df[df["experiment_type"] != "2A3_MaP"]
signal_to_noise_mask = (df1["SN_filter"] == 1).values
x = np.array(df1[signal_to_noise_mask].iloc[:, 1])
y_1 = np.array(df1[signal_to_noise_mask].iloc[:, 7:213])
y_2 = np.array(df2[signal_to_noise_mask].iloc[:, 7:213])

# Cut reactivity to [0,1]
y_1 = np.clip(y_1, 0, 1)
y_2 = np.clip(y_2, 0, 1)

# Transform RNA sequences to embeddings
print("Transforming RNA sequences to embeddings...")
data = []
for i in range(len(x)):
    data.append((str(i), x[i]))
labels, strs, tokens = batch_converter(data)

# Cut to N samples
tokens = tokens[START:END]
y_1 = y_1[START:END]
y_2 = y_2[START:END]

# Saving tokenized sequences and labels
print("Saving tokenized sequences...")
if not os.path.exists("../data/sequences"):
    os.makedirs("../data/sequences")
if not os.path.exists("../data/2A3_MaP"):
    os.makedirs("../data/2A3_MaP")
if not os.path.exists("../data/DMS_MaP"):
    os.makedirs("../data/DMS_MaP")

for i in range(START, END):
    np.savez("../data/sequences/{}.npz".format(i), x=tokens[i - START])
    np.savez("../data/2A3_MaP/{}.npz".format(i), x=np.concatenate([[np.nan], y_1[i - START], [np.nan]]))
    np.savez("../data/DMS_MaP/{}.npz".format(i), x=np.concatenate([[np.nan], y_2[i - START], [np.nan]]))

# Extract embeddings
if not os.path.exists("../data/fm_embeddings"):
    os.makedirs("../data/fm_embeddings")

print("Extracting embeddings...")
batch_size = 128
for idx in range(int(np.ceil((END-START)/batch_size))):
    print("Batch {} of {}".format(idx+1, int(np.ceil((END-START)/batch_size))))

    batch = tokens[idx*batch_size:(idx+1)*batch_size,:]
    batch = batch.to(device)
    with torch.no_grad():
        results = model(batch, repr_layers=[12])
    embeddings = results["representations"][12].cpu().numpy()

    del batch, results
    torch.cuda.empty_cache()

    for i in range(embeddings.shape[0]):
        np.savez("../data/fm_embeddings/{}.npz".format(idx*batch_size + i + START), x=embeddings[i])
