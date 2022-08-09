import pytest
import torch

model_size = 20
key_size = 2
value_size = 10
num_heads = 8
feedforward_size = 7
num_queries = 3
num_values = num_queries
batch_size = 5

def test_import():
    from attention_network.model.modules.EncoderStack import EncoderStack
 
def test_module_init():
    from attention_network.model.modules.EncoderStack import EncoderStack

    network = EncoderStack(model_size, key_size, value_size, num_heads, feedforward_size)
    assert(network)

def test_output_sizes():
    from attention_network.model.modules.EncoderStack import EncoderStack

    embedding = torch.ones(size=(batch_size, num_queries, model_size), dtype=torch.float)
    
    network = EncoderStack(model_size, key_size, value_size, num_heads, feedforward_size)
    output_embedding, attention_weights = network(embedding)

    assert(tuple(output_embedding.shape) == (batch_size, num_queries, model_size))
    assert(tuple(attention_weights.shape) == (batch_size, num_queries, num_values * num_heads))

