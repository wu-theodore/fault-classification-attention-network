import pytest
import torch

num_queries = 2
batch_size = 5
model_size = 20

def test_import():
    from attention_network.model.modules.ResidualAddAndNorm import ResidualAddAndNorm
 
def test_module_init():
    from attention_network.model.modules.ResidualAddAndNorm import ResidualAddAndNorm

    network = ResidualAddAndNorm(model_size)
    assert(network)

def test_output_sizes():
    from attention_network.model.modules.ResidualAddAndNorm import ResidualAddAndNorm

    new = torch.randn(batch_size, num_queries, model_size)
    residual = torch.ones(batch_size, num_queries, model_size)
    
    network = ResidualAddAndNorm(model_size)
    output = network(residual, new)

    assert(tuple(output.shape) == (batch_size, num_queries, model_size))

