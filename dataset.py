import torch
from torch.utils.data import Dataset
import numpy as np
import pickle as pkl
from utils import *

class CopeDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]

        return torch.Tensor(sample), torch.Tensor(label)