import torch

class PositionalEncoding:
    def __init__(self, model_size):
        self.model_size = model_size

    def generate_encoding(self, max_seq_len=1000):
        """
        Creates the sinusoidal positional encoding that is added to inputs.

        Arguments:
            max_seq_len: The maximum sequence length expected to be encountered.

        Returns:
            pos_encodings: Positional encodings with shape (max_seq_len, model_size)
        """
        pos_indices = torch.arange(max_seq_len)[..., None]
        dim_indices = torch.arange(self.model_size // 2)[None, ...]
        exponents = (2 * dim_indices).float() / (self.model_size)
        trig_args = pos_indices / (10000**exponents)
        sin_terms = torch.sin(trig_args)
        cos_terms = torch.cos(trig_args)

        pos_encodings = torch.zeros((max_seq_len, self.model_size), dtype=torch.float32)
        pos_encodings[:, 0::2] = sin_terms
        pos_encodings[:, 1::2] = cos_terms

        return pos_encodings