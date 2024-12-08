import torch
import torch.nn as nn

class LayerNormalizationLayer(nn.Module):

    def __init__(self, features: int =1, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))  # Learnable parameter, shape: (features)
        self.bias = nn.Parameter(torch.zeros(features))  # Learnable parameter, shape: (features)

    def forward(self, x):
        # Input shape: (batch_size, seq_len, hidden_size)
        
        mean = x.mean(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        # Mean calculated along the last dimension (hidden_size), broadcasting for other dimensions.
        
        std = x.std(dim=-1, keepdim=True)  # (batch_size, seq_len, 1)
        # Standard deviation calculated along the last dimension (hidden_size), broadcasting for other dimensions.
        
        # Output shape: (batch_size, seq_len, hidden_size)
        # Apply layer normalization and return the result
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
