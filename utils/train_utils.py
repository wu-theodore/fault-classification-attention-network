import torch

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from model.AttentionNetwork import AttentionNetwork
from model.RNNBaseline import RNNBaseline
from model.DNNBaseline import DNNBaseline
from model.CNNBaseline import CNNBaseline
from sklearn.model_selection import KFold
from utils.CAVSignalDataset import CAVSignalDataset
from utils.Transforms import MinMaxScale

def load_data(data_dir, batch_size=None, shuffle=True, num_folds=5, transform=MinMaxScale(), channel_first=False):
    dataset = CAVSignalDataset(data_dir, transform=transform, channel_first=channel_first)
    dataset_size = len(dataset)
    print(f"Dataset has {dataset_size} samples total.")
    kfold = KFold(n_splits=num_folds, shuffle=True)
    print(f"Splitting dataset with {int(dataset_size * (1 - 1/num_folds))} samples in train and {int(dataset_size * (1/num_folds))} samples in val")
    
    data_loaders = []
    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        
        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        if not batch_size:
            train_loader = DataLoader(dataset, batch_size=len(train_idx), sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=len(val_idx), sampler=val_sampler)
        else:
            train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler)
        
        data_loaders.append((train_loader, val_loader))

    return data_loaders

def load_model(config, device):
    model_type = config["model"]
    if model_type == "attention":
        model = AttentionNetwork(config, device)
    elif model_type == "rnn":
        model = RNNBaseline(config, device)
    elif model_type == "dnn":
        model = DNNBaseline(config, device)
    elif model_type == "cnn":
        model = CNNBaseline(config, device)
    else:
        raise ValueError("Incorrect model passed.")
    model.to(device)
    return model

def compute_batch_accuracy(pred, label):
    num_preds = len(label)
    pred = torch.argmax(pred, dim=1)
    batch_accuracy = (torch.sum(pred == label) / num_preds).item()
    return batch_accuracy
    