import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.device = device

        self.embedding_layer = nn.Linear(self.config["state_size"], self.config["embedding_size"])
        self.RNN = nn.LSTM(self.config["embedding_size"], self.config["model_size"], num_layers=self.config["num_layers"], 
            dropout=self.config["dropout"])
        self.hidden_layer = nn.Linear(self.config["model_size"], self.config["hidden_layer_size"])
        self.classification_layer = nn.Linear(self.config["hidden_layer_size"], self.config["num_classes"])
        self.dropout = nn.Dropout(self.config["dropout"])
        self.relu = nn.ReLU()

    def forward(self, input):
        embedding = self.embedding_layer(input)
        output, (h_n, c_n) = self.RNN(embedding.transpose(0, 1))

        hidden_state = self.dropout(self.relu(self.hidden_layer(output[-1])))
        logits = self.classification_layer(hidden_state)

        return logits

        
