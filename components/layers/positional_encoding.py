import torch
import torch.nn as nn
import math


class PositionalEncodingBase(nn.Module):
    def __init__(self, model_dimension: int, sequence_length: int):
        super(PositionalEncodingBase, self).__init__()
        self.model_dimension = model_dimension
        self.sequence_length = sequence_length

    def generate_positional_encoding(self) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")


class SinusoidalPositionalEncoding(PositionalEncodingBase):
    def __init__(self, model_dimension: int, sequence_length: int):
        super(SinusoidalPositionalEncoding, self).__init__(model_dimension, sequence_length)

    def generate_positional_encoding(self) -> torch.Tensor:
        encoding = torch.zeros(self.sequence_length, self.model_dimension)

        # Compute the position indices and reshape them
        positions = torch.arange(self.sequence_length, dtype=torch.float).unsqueeze(1)  # Shape: (sequence_length, 1)

        # Compute frequency terms for the sine and cosine functions
        frequency_terms = self._compute_frequency_terms()

        # Apply sine to even indices and cosine to odd indices
        encoding[:, 0::2] = torch.sin(positions * frequency_terms)  # Even indices
        encoding[:, 1::2] = torch.cos(positions * frequency_terms)  # Odd indices

        # Add a batch dimension to make the encoding ready for input addition
        return encoding.unsqueeze(0)  # Shape: (1, sequence_length, model_dimension)

    def _compute_frequency_terms(self) -> torch.Tensor:
        scaling_factor = -math.log(10000.0) / self.model_dimension
        even_indices = torch.arange(0, self.model_dimension, 2, dtype=torch.float)
        return torch.exp(even_indices * scaling_factor)


class PositionalEncodingLayer(nn.Module):
    def __init__(self, 
                 model_dimension: int, 
                 sequence_length: int, 
                 dropout_rate: float) -> None:
        super(PositionalEncodingLayer, self).__init__()
        
        # Dropout layer for regularization
        self.dropout_layer = nn.Dropout(dropout_rate)
        
        # Directly instantiate the SinusoidalPositionalEncoding class
        self.positional_encoding_generator = SinusoidalPositionalEncoding(model_dimension,
                                                                          sequence_length)
        
        # Generate the positional encoding matrix
        self.positional_encoding_matrix = self.positional_encoding_generator.generate_positional_encoding()

        # Register positional encoding as a buffer to prevent it from being treated as a model parameter
        self.register_buffer("positional_encoding", self.positional_encoding_matrix)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        # Slice the positional encoding to match the input sequence length
        positional_encoding = self.positional_encoding[:, :input_tensor.size(1), :]

        # Add positional encoding to the input tensor
        input_with_encoding = input_tensor + positional_encoding.requires_grad_(False)

        # Apply dropout and return the result
        return self.dropout_layer(input_with_encoding)
