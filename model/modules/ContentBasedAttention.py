import torch
import torch.nn as nn

class ContentBasedAttention(nn.Module):
    def __init__(self, h_size, s_size, cv_size):
        super(ContentBasedAttention, self).__init__()

        self.weight_layer = nn.Linear(h_size + s_size, cv_size)
        self.score_layer = nn.Linear(cv_size, 1)

        self.tanh = nn.Tanh()

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h, s):
        """
        Forward pass for Content-based attention. 

        Arguments:
            h: current hidden state, shape (batch_size x seq_len x lstm_size)
            s: previous hidden state, shape (batch_size x seq_len x conv3_size)
        Returns:
            cv: context vector from computing attention on h and s, shape (batch_size x lstm_size)
        """
        # Compute attention weights
        attention_weights = self.score_layer(self.tanh(self.weight_layer(torch.cat((h, s), dim=-1))))
        normalized_weights = self.softmax(attention_weights)

        cv = normalized_weights.transpose(1, 2) @ h
        cv = cv.squeeze(dim=1)
        return cv

