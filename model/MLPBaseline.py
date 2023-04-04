import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBaseline(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config

        # Unpack config
        num_features = self.config["num_features"]
        num_classes = self.config["num_classes"]
        self.hidden_size = self.config["hidden_size"]

        self.device = device

        self.hidden_layer_1 = nn.Linear(num_features, num_features)
        self.hidden_layer_2 = nn.Linear(num_features, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, num_classes)

    def forward(self, input):
        x = F.relu(self.hidden_layer_1(input))
        x = F.relu(self.hidden_layer_2(x))
        
        logits = self.output_layer(x)

        return logits
