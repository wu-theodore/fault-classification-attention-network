import pytest
import torch

state_size = 15
key_size = 2
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
    "key_size": key_size,
    "value_size": value_size,
    "num_heads": num_heads,
    "feedforward_size": feedforward_size,
    "num_encoders": num_encoders,
    "num_classes": num_classes
}

def test_import():
    from attention_network.model.AttentionNetwork import AttentionNetwork
 
def test_module_init():
    from attention_network.model.AttentionNetwork import AttentionNetwork

    network = AttentionNetwork(config)
    assert(network)

def test_output_sizes():
    from attention_network.model.AttentionNetwork import AttentionNetwork

    embedding = torch.ones(size=(batch_size, num_queries, state_size), dtype=torch.float)
    
    network = AttentionNetwork(config)
    output_logits, embeddings, attention_weights = network(embedding)

    assert(tuple(output_logits.shape) == (batch_size, num_classes))
    assert(tuple(attention_weights.shape) == (num_encoders, batch_size, num_queries, num_queries * num_heads))
    assert(tuple(embeddings.shape) == (num_encoders + 1, batch_size, num_queries, model_size))

