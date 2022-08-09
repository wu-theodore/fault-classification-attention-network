import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    def __init__(self, key_size, value_size, model_size):
        super(ScaledDotProductAttention, self).__init__()

        self.model_size = model_size        # d_model
        self.key_size = key_size            # d_k
        self.value_size = value_size        # d_v

        # Projections for query, key, and value tensors.
        self.W_Q = nn.Linear(self.model_size, self.key_size)
        self.W_K = nn.Linear(self.model_size, self.key_size)
        self.W_V = nn.Linear(self.model_size, self.value_size)

        self.softmax = nn.Softmax(dim=2)
        self.scaling_factor = torch.rsqrt(
            torch.tensor(self.key_size, dtype=torch.float)
        )

    def forward(self, Q, K, V):
        """
        Forward pass for ScaledDotProduct attention. 

        Arguments:
            Q: Tensor of shape (batch_size x n_q x d_model)
            K: Tensor of shape (batch_size x n_v x d_model)
            V: Tensor of shape (batch_size x n_v x d_model) 
        Returns:
            attention_weights: Tensor of shape (batch_size x n_q x n_v)
            context: Tensor of shape (batch_size x n_q x d_v)
        """
        queries = self.W_Q(Q)
        values = self.W_V(V)
        keys = self.W_K(K)
        
        unnormalized_attention = queries @ keys.transpose(2, 1) * self.scaling_factor
        attention_weights = self.softmax(unnormalized_attention)
        
        context = attention_weights @ values
        return context, attention_weights
