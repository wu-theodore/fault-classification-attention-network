import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Unpack config
        state_size = self.config["state_size"]
        model_size = self.config["model_size"]
        hidden_size = self.config["value_size"]
        num_encoders = self.config["num_encoders"]
        num_classes = self.config["num_classes"]

        self.input_embedding = nn.Linear(state_size, model_size)
        self.RNN = nn.LSTM(model_size, hidden_size, num_encoders, batch_first=True)
        self.classification_layer = nn.Linear(num_encoders * hidden_size, num_classes)

    def forward(self, input):
        embedding = self.input_embedding(input)
        output, (h_n, c_n) = self.RNN(embedding)

        h_n = torch.transpose(h_n, 0, 1)
        batch_size = h_n.shape[0]
        encoder_state = h_n.reshape(batch_size, -1)

        logits = self.classification_layer(encoder_state)

        return logits

        
