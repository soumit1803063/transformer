import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    def __init__(self, model_dimension: int, feedforward_dimension: int, dropout_rate: float) -> None:
        super(FeedForwardBlock, self).__init__()
        
        # Define the layers
        self.dense_layer_1 = nn.Linear(model_dimension, feedforward_dimension)  # w1 and b1
        self.dropout_layer = nn.Dropout(dropout_rate)
        self.dense_layer_2 = nn.Linear(feedforward_dimension, model_dimension)  # w2 and b2

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Input: (batch_size, sequence_length, model_dimension)
        # Output: (batch_size, sequence_length, model_dimension)
        
        # Apply the feedforward network
        x = torch.relu(self.dense_layer_1(input_tensor))  # (batch_size, sequence_length, feedforward_dimension)
        x = self.dropout_layer(x)
        output_tensor = self.dense_layer_2(x)  # (batch_size, sequence_length, model_dimension)
        
        return output_tensor
