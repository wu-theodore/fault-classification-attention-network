import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.ContentBasedAttention import ContentBasedAttention

class MSALSTMCNNBaseline(nn.Module):
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

        self.rnn = nn.LSTM(self.config["conv3_filters"], self.config["lstm_size"], batch_first=True)
        
        self.attention = ContentBasedAttention(config["lstm_size"], config["conv3_filters"], config["cv_size"])

        self.classification_layer = nn.Linear(config["lstm_size"], config["num_classes"])

        self.dropout = nn.Dropout(self.config["dropout"])
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        x = self.dropout(self.relu(self.conv1(input)))
        x = self.dropout(self.relu(self.conv2(x)))
        x = self.dropout(self.relu(self.conv3(x)))

        feature_seq = self.pool(x)
        feature_seq = feature_seq.transpose(1, 2)

        attention_context, _ = self.rnn(feature_seq)
        context_vector = self.attention(attention_context, feature_seq)
        
        logits = self.classification_layer(context_vector)
        return logits
