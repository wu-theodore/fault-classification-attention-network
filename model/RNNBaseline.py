import torch
import torch.nn as nn

class RNNBaseline(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config

        # Unpack config
        state_size = self.config["state_size"]
        model_size = self.config["model_size"]
        num_encoders = self.config["num_encoders"]
        num_classes = self.config["num_classes"]
        self.hidden_size = self.config["value_size"]

        self.device = device
        self.input_embedding = nn.Linear(state_size, model_size)
        self.RNN = nn.LSTM(model_size, self.hidden_size, num_layers=num_encoders, 
            batch_first=True, bidirectional=True)
        self.classification_layer = nn.Linear(2*self.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        embedding = self.input_embedding(input)
        output, (h_n, c_n) = self.RNN(embedding)

        output_forward = output[:, -1, :self.hidden_size]
        output_reverse = output[:, 0, self.hidden_size:]
        stacked_encoder_state = torch.cat((output_forward, output_reverse), dim=1)

        logits = self.classification_layer(stacked_encoder_state)
        normalized_logits = self.softmax(logits)

        return normalized_logits

        
