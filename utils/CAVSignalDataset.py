import os 
import torch
import numpy as np
from torch.utils.data import Dataset

class CAVSignalDataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        self.labels = self.get_labels(data_dir)
        self.data_dir = data_dir
        self.transform = transform
        self.target_transform = target_transform
        self.label_map = {
            "actuator": 0,
            "attack": 1,
            "delay": 2,
            "distracted": 3,
            "drunk": 4,
        }

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        file, label = self.labels[index]
        file_path = os.path.join(self.data_dir, label, file)
        data = np.loadtxt(file_path, delimiter=',').T
        label = self.label_map[label]
        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            label = self.target_transform(label)
        data = torch.tensor(data, dtype=torch.float64)
        return data, label

    def get_labels(self, data_dir):
        labels = []
        for label in os.listdir(data_dir):
            labels.extend([
                (sample, label)
                for sample in os.listdir(os.path.join(data_dir, label))
            ])
        return labels
        