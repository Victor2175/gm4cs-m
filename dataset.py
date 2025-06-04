import torch
from torch.utils.data import Dataset

class CopeDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = torch.tensor(samples).to(torch.float32)
        self.labels = torch.tensor(labels).to(torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        label = self.labels[index]

        return sample, label