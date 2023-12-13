import fm
import numpy as np
import os
import pandas as pd
import torch
import tqdm

# Load and order data
print("Loading data...")
df = pd.read_csv("../../../csv/train_data.csv")
df = df[df["experiment_type"] == "2A3_MaP"]
signal_to_noise_mask = (df["SN_filter"] == 1).values
df = df[signal_to_noise_mask]
df = df.reset_index()

# Load secondary data
print("Loading secondary data...")
file_list = os.listdir("../../../csv/secondary")
file_list = [file for file in file_list if file.endswith(".csv")]

# Find indices of secondary data
secondary_data = []
for file in file_list:
    print("Loading secondary data from " + file + "...")
    secondary = pd.read_csv("../../../csv/secondary/" + file)
    for i in tqdm.tqdm(range(len(secondary))):
        seq = secondary["sequence"][i]
        idx = df[df["sequence"] == seq]
        if idx.empty:
            continue
        idx = idx.index[0]
        data = secondary["vienna2_mfe"][i]
        binary = [np.nan]
        for char in data:
            if char == ".":
                binary.append(0)
            else:
                binary.append(1)
                
        while len(binary) < 208:
            binary.append(np.nan)

        secondary_data.append([idx, binary])

# Saving secondary structure data
if not os.path.exists("../data/secondary"):
    os.makedirs("../data/secondary")

for sample in secondary_data:
    idx = sample[0]
    binary = np.array(sample[1])
    np.savez("../data/secondary/{}.npz".format(idx), x=binary)
