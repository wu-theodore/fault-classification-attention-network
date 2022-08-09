import pytest
import torch

key_size = 2
value_size = 10
num_queries = 1
num_values = 3
batch_size = 5
model_size = 20
num_heads = 8

def test_import():
    from attention_network.model.modules.MultiHeadAttention import MultiHeadAttention
 
def test_module_init():
    from attention_network.model.modules.MultiHeadAttention import MultiHeadAttention

    layer = MultiHeadAttention(model_size, key_size, value_size, num_heads)
    assert(len(layer.attentions) == num_heads)

def test_output_sizes():
    from attention_network.model.modules.MultiHeadAttention import MultiHeadAttention

    queries = torch.ones(size=(batch_size, num_queries, model_size), dtype=torch.float)
    keys = torch.ones(size=(batch_size, num_values, model_size), dtype=torch.float)
    values = torch.ones(size=(batch_size, num_values, model_size), dtype=torch.float)
    
    layer = MultiHeadAttention(model_size, key_size, value_size, num_heads)
    context, attention_weights = layer(queries, keys, values)

    assert(tuple(context.shape) == (batch_size, num_queries, model_size))
    assert(tuple(attention_weights.shape) == (batch_size, num_queries, num_values * num_heads))

