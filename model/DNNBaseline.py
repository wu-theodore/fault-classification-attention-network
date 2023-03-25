import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNBaseline(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config

        # Unpack config
        num_features = self.config["num_features"]
        num_classes = self.config["num_classes"]
        self.hidden_size = self.config["hidden_size"]

        self.device = device

        self.hidden_layer_1 = nn.Linear(num_features, num_features)
        self.hidden_layer_2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.output_layer = nn.Linear(self.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = F.relu(self.hidden_layer_1(input))
        x = F.relu(self.hidden_layer_2(x))
        
        output = self.output_layer(x)
        logits = self.softmax(output)

        return logits
