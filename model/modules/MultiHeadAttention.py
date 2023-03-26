import torch
import torch.nn as nn
from .ScaledDotProductAttention import ScaledDotProductAttention

class MultiHeadAttention(nn.Module):
    def __init__(self, model_size, key_size, value_size, num_heads, dropout):
        super(MultiHeadAttention, self).__init__()

        # Configurable hyperparameters
        self.h = num_heads                  # h
        self.model_size = model_size        # d_model
        self.key_size = key_size            # d_k
        self.value_size = value_size        # d_v

        self.dropout = dropout

        # Multi-head attention layer.
        self.attentions = nn.ModuleList(
            [ScaledDotProductAttention(self.key_size, self.value_size, self.model_size, self.dropout) for i in range(self.h)]
        )

        # Final projection layer back to d_model
        self.W_M = nn.Linear(self.h * self.value_size, self.model_size)
        

    def forward(self, Q, K, V):
        """
        Forward pass for multi-head attention.
    
        Arguments:
            Q: Query tensor of shape (batch_size x n_q x d_model)
            K: Key tensor of shape (batch_size x n_v x d_model)
            V: Value tensor of shape (batch_size x n_v x d_model)
        Returns:
            stacked_attention_weights: Stacked attention weights returned by each head of the attention modules, length == h.        
            multi_head: Output tensor of the multi-head attention layer (batch_size x n_q x d_model)
        """

        attention_weights_list = []
        context_list = []
        for i in range(self.h):
            context, attention_weights = self.attentions[i](Q, K, V)
            context_list.append(context)
            attention_weights_list.append(attention_weights)

        stacked_attention_weights = torch.cat(attention_weights_list, dim=-1)
        stacked_context = torch.cat(context_list, dim=-1)
        multi_head = self.W_M(stacked_context)

        return multi_head, stacked_attention_weights

