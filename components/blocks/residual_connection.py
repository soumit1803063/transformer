import torch
import torch.nn as nn
from components.layers import LayerNormalizationLayer


class ResidualBlock(nn.Module):
    def __init__(self, features: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalizationLayer(features)

    def forward(self, x, sublayer):
        normalized_x = self.norm(x)  # Apply layer normalization
        # normalized_x: (batch_size, seq_len, features)
        
        sublayer_output = sublayer(normalized_x)  # Pass through the sublayer
        # sublayer_output: (batch_size, seq_len, features)
        
        dropped_output = self.dropout(sublayer_output)  # Apply dropout
        # dropped_output: (batch_size, seq_len, features)
        
        return x + dropped_output  # Add residual connection
        # Output: (batch_size, seq_len, features)
