import pytest
import torch

num_queries = 2
batch_size = 5
model_size = 20
feedforward_size = 100

def test_import():
    from attention_network.model.modules.PositionalFF import PositionalFeedForward
 
def test_module_init():
    from attention_network.model.modules.PositionalFF import PositionalFeedForward

    network = PositionalFeedForward(model_size, feedforward_size)
    assert(network)

def test_output_sizes():
    from attention_network.model.modules.PositionalFF import PositionalFeedForward

    input = torch.ones(batch_size, num_queries, model_size)
    
    network = PositionalFeedForward(model_size, feedforward_size)
    output = network(input)

    assert(tuple(output.shape) == (batch_size, num_queries, model_size))

