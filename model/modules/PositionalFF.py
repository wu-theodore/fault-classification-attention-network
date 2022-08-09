import torch
import torch.nn as nn

class PositionalFeedForward(nn.Module):
    def __init__(self, model_size, feedforward_size):
        super(PositionalFeedForward, self).__init__()

        self.model_size = model_size
        self.ff_size = feedforward_size

        self.network = nn.Sequential(
            nn.Linear(self.model_size, self.ff_size),
            nn.ReLU(),
            nn.Linear(self.ff_size, self.model_size)
        )

    def forward(self, input):
        """
        Forward pass of a simple positional feedforward network. Applies a 
        fully connected layer with ReLU activation, followed by another fully
        connected layer to project the tensor back to the model dimensionality.

        Arguments:
            input: Input tensor of shape (batch_size x n_q x d_model)

        Returns:
            output: Output tensor of shape (batch_size x n_q x d_model)
        """
        output = self.network(input)
        return output