import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("../../../../csv/train_data.csv")
mask = df["experiment_type"] == "2A3_MaP"
shape1 = np.array(df[mask].iloc[:, 7:213])
shape1 = shape1.flatten()
shape1 = shape1[~np.isnan(shape1)]
shape1 = np.clip(shape1, 0, 1)

shape2 = np.array(df[~mask].iloc[:, 7:213])
shape2 = shape2.flatten()
shape2 = shape2[~np.isnan(shape2)]
shape2 = np.clip(shape2, 0, 1)

plt.hist(shape1, bins=10, edgecolor='black', density=True, alpha=0.5, label='2A3_MaP')
plt.hist(shape2, bins=10, edgecolor='black', density=True, alpha=0.5, label='DMS_MaP')
plt.legend()
plt.xlabel('SHAPE value')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('../../outputs/figures/data_hist.png')

