import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNBaseline(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        
        self.device = device

        self.conv1 = nn.Conv1d(self.config["num_vehicles"], self.config["conv1_filters"], 
            self.config["kernel_size"], self.config["stride"])
        self.conv2 = nn.Conv1d(self.config["conv1_filters"], self.config["conv2_filters"],
            self.config["kernel_size"], self.config["stride"])
        self.conv3 = nn.Conv1d(self.config["conv2_filters"], self.config["conv3_filters"],
            self.config["kernel_size"], self.config["stride"])

        self.pool = nn.MaxPool1d(self.config["pool_size"])

        self.fc1 = nn.Linear(self.config["conv3_filters"], self.config["hidden_layer_size"])
        self.fc2 = nn.Linear(self.config["hidden_layer_size"], self.config["num_classes"])

        self.dropout = nn.Dropout(self.config["dropout"])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.pool(self.relu(self.conv1(input)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = torch.mean(x, dim=-1)
        x = self.dropout(self.relu(self.fc1(x)))
        logits = self.fc2(x)

        return logits
