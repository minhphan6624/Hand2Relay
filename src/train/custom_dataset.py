import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class HandGestureDataset(Dataset):
    """Custom Dataset for loading hand gesture data from CSV files."""
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        # Extract features and labels
        self.data = self.data.iloc[:, :-1].values.astype(np.float32) 
        self.labels = self.data.iloc[:, -1].values.astype(np.int64) 

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])