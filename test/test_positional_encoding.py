import pytest

max_seq_len = 100
model_size = 20
    

def test_import():
    from attention_network.model.modules.PositionalEncoding import PositionalEncoding
 
def test_module_init():
    from attention_network.model.modules.PositionalEncoding import PositionalEncoding

    layer = PositionalEncoding(model_size)
    assert(layer)

def test_output_sizes():
    from attention_network.model.modules.PositionalEncoding import PositionalEncoding

    layer = PositionalEncoding(model_size)

    encoding = layer.generate_encoding(max_seq_len=max_seq_len)

    assert(encoding.shape == (max_seq_len, model_size))

