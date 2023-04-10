import torch
import torch.nn as nn
import torch.nn.functional as F

from model.modules.PositionalEncoding import PositionalEncoding
from model.modules.EncoderStack import EncoderStack

class AttentionNetwork(nn.Module):
    def __init__(self, config, device="cuda"):
        super().__init__()

        self.config = config

        # Unpack config
        state_size = config["num_vehicles"]
        model_size = config["model_size"]
        value_size = config["value_size"]
        #key_size = config["key_size"]
        key_size = value_size
        num_heads = config["num_heads"]
        feedforward_size = config["feedforward_size"]
        num_encoders = config["num_encoders"]
        num_classes = config["num_classes"]
        dropout = config["dropout"]

        embedding = config["embedding_type"]

        self.device = device
        if embedding == "cnn":
            self.input_embedding = nn.Conv1d(state_size, model_size, kernel_size=10)
        else:
            self.input_embedding = nn.Linear(state_size, model_size)
        
        self.positional_encoding = PositionalEncoding(model_size)

        self.encoder_stacks = nn.ModuleList(
            [
                EncoderStack(model_size, key_size, value_size, num_heads, feedforward_size, dropout)
                for n in range(num_encoders)
            ]
        )
        self.classification_layer = nn.Linear(model_size, num_classes)


    def forward(self, input):
        """
        Forward pass for the attention network.
    
        Arguments:
            input: Input states of shape (batch_size x seq_len x state_size)
        Returns:
            output: Output classification logits of shape (batch_size x num_classes)
            stacked_embeddings: The output embeddings at each encoder layer in the network.
            stacked_attention_weights: The attention weights for each encoder layer in the network.
        """        
        embedding = self.input_embedding(input)
        if self.config["embedding_type"] == "cnn":
            embedding = embedding.transpose(1, 2)
        pos_encoding = self.positional_encoding.generate_encoding()[:embedding.shape[1]]
        pos_encoding = pos_encoding.to(self.device)
        embedding = embedding + pos_encoding

        # Keep track of states throughout the model.
        attention_weights_list = []
        embedding_list = [embedding]

        for i in range(len(self.encoder_stacks)):
            embedding, attention_weights = self.encoder_stacks[i](embedding)
            embedding_list.append(embedding)
            attention_weights_list.append(attention_weights)

        stacked_attention_weights = torch.stack(attention_weights_list)
        stacked_embeddings = torch.stack(embedding_list)

        # reduced_embedding = torch.mean(embedding, dim=1)
        reduced_embedding = embedding[:, 0]
        logits = self.classification_layer(reduced_embedding)

        return logits, stacked_attention_weights, stacked_embeddings

