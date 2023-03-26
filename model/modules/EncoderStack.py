import torch
import torch.nn as nn

from model.modules.MultiHeadAttention import MultiHeadAttention
from model.modules.PositionalFF import PositionalFeedForward
from model.modules.ResidualAddAndNorm import ResidualAddAndNorm

class EncoderStack(nn.Module):
    def __init__(self, model_size, key_size, value_size, num_heads, feedforward_size, dropout):
        super().__init__()

        self.multi_head_attention = MultiHeadAttention(model_size, key_size, value_size, num_heads, dropout)
        self.positional_ff = PositionalFeedForward(model_size, feedforward_size)

        self.residual = ResidualAddAndNorm(model_size)

    def forward(self, E):
        """
        Forward pass for a single encoder stack. 
    
        Arguments:
            E: Input state embedding of shape (batch_size x seq_len x model_size)
        Returns:
            E_out: Output state embedding of shape (batch_size x seq_len x model_size)
            attention_weights: Attention weights of the multi-head attention layer.
        """
        context, attention_weights = self.multi_head_attention(E, E, E)
        normalized_context = self.residual(E, context)
        
        feedforward_context = self.positional_ff(normalized_context)
        E_out = self.residual(normalized_context, feedforward_context)

        return E_out, attention_weights