import torch
import torch.nn as nn

class ResidualAddAndNorm(nn.Module):
    def __init__(self, model_size):
        super(ResidualAddAndNorm, self).__init__()

        self.layer_norm = nn.LayerNorm(model_size)

    def forward(self, residual, new):
        """
        Forward pass of a residual network connection. Applies Layer-
        Normalization to the summed residual.

        Arguments:
            residual: Tensor corresponding to the untransformed residual
            new: Tensor corresponding to the residual after being transformed by some neural network. 
                 Must be the same size as the residual.

        Returns:
            output: Tensor corresponding to the Add & Norm output of the residual connection.
        """

        summed_residual = residual + new
        output = self.layer_norm(summed_residual)
        return output

