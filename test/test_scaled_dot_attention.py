import pytest
import torch
import math

key_size = 2
value_size = 10
num_queries = 1
num_values = 3
batch_size = 5
model_size = 20

def test_import():
    from attention_network.model.modules.ScaledDotProductAttention import ScaledDotProductAttention
 
def test_module_init():
    from attention_network.model.modules.ScaledDotProductAttention import ScaledDotProductAttention

    key_size = 100
    value_size = 100
    model_size = 100
    
    layer = ScaledDotProductAttention(key_size, value_size, model_size)
    assert(round(float(layer.scaling_factor), 3) == 1 / math.sqrt(key_size))

def test_output_sizes():
    from attention_network.model.modules.ScaledDotProductAttention import ScaledDotProductAttention

    queries = torch.ones(size=(batch_size, num_queries, model_size), dtype=torch.float)
    keys = torch.ones(size=(batch_size, num_values, model_size), dtype=torch.float)
    values = torch.ones(size=(batch_size, num_values, model_size), dtype=torch.float)
    
    layer = ScaledDotProductAttention(key_size, value_size, model_size)

    context, attention_weights = layer(queries, keys, values)

    assert(tuple(layer.W_Q(queries).shape) == (batch_size, num_queries, key_size))
    assert(tuple(context.shape) == (batch_size, num_queries, value_size))
    assert(tuple(attention_weights.shape) == (batch_size, num_queries, num_values))

