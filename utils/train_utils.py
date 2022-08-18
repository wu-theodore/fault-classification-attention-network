import torch

import torch
from torch.utils.data import DataLoader, random_split

from model.AttentionNetwork import AttentionNetwork
from model.RNNBaseline import RNNBaseline
from utils.CAVSignalDataset import CAVSignalDataset
from utils.Transforms import MinMaxScale

def load_data(data_dir, train_split, batch_size=100, shuffle=True):
    dataset = CAVSignalDataset(data_dir, transform=MinMaxScale())
    dataset_size = len(dataset)
    split_size = int(dataset_size * train_split)
    print(f"Splitting dataset with {split_size} samples in train and {dataset_size - split_size} samples in val")
    train_dataset, val_dataset = random_split(dataset, 
        [split_size, dataset_size - split_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

def load_model(config, device):
    model_type = config["model"]
    if model_type == "attention":
        model = AttentionNetwork(config, device)
    elif model_type == "rnn":
        model = RNNBaseline(config, device)
    else:
        raise ValueError("Incorrect model passed.")
    model.to(device)
    return model

def compute_batch_accuracy(pred, label):
    num_preds = len(label)
    pred = torch.argmax(pred, dim=1)
    batch_accuracy = (torch.sum(pred == label) / num_preds).item()
    return batch_accuracy
    