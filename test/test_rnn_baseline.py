import pytest
import torch

state_size = 15
value_size = 10
batch_size = 5
model_size = 20
num_heads = 8
feedforward_size = 7
num_encoders = 3
num_classes = 4
num_queries = 30

config = {
    "state_size": state_size,
    "model_size": model_size,
    "value_size": value_size,
    "num_encoders": num_encoders,
    "num_classes": num_classes,
    "dropout": 0.2
}

def test_import():
    from attention_network.model.RNNBaseline import RNNBaseline
 
def test_module_init():
    from attention_network.model.RNNBaseline import RNNBaseline

    network = RNNBaseline(config)
    assert(network)

def test_output_sizes():
    from attention_network.model.RNNBaseline import RNNBaseline

    embedding = torch.ones(size=(batch_size, num_queries, state_size), dtype=torch.float)
    
    network = RNNBaseline(config)
    output_logits= network(embedding)

    assert(tuple(output_logits.shape) == (batch_size, num_classes))

