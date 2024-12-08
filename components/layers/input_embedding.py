import torch
import torch.nn as nn

class InputEmbeddingLayer(nn.Module):
    def __init__(self,
                 model_dimension: int,
                 vocab_size: int):
        super().__init__()
        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding_layer = nn.Embedding(vocab_size, model_dimension)
    
    def forward(self, x):
        # Input shape: (batch_size, sequence_length)
        x = self.embedding_layer(x)
        # Output shape after embedding: (batch_size, sequence_length, model_dimension)
        
        x *= torch.sqrt(torch.tensor(self.model_dimension, dtype=x.dtype, device=x.device))
        # Output shape after scaling: (batch_size, sequence_length, model_dimension)
        
        return x
