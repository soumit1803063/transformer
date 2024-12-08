import torch
import torch.nn as nn
import math

class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, model_dimension: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.model_dimension = model_dimension  # Embedding vector size
        self.num_heads = num_heads  # Number of attention heads
        # Ensure model_dimension is divisible by num_heads
        assert model_dimension % num_heads == 0, "model_dimension must be divisible by num_heads"

        self.head_dim = model_dimension // num_heads  # Dimension of vector for each head
        self.query_proj = nn.Linear(model_dimension, model_dimension, bias=False)  # Query projection
        self.key_proj = nn.Linear(model_dimension, model_dimension, bias=False)  # Key projection
        self.value_proj = nn.Linear(model_dimension, model_dimension, bias=False)  # Value projection
        self.output_proj = nn.Linear(model_dimension, model_dimension, bias=False)  # Output projection
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def scaled_dot_product_attention(query, key, value, mask, dropout: nn.Dropout):
        # (batch, num_heads, seq_len, head_dim)-> (head_dim)
        head_dim = query.shape[-1]
        # Calculate scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)-->
        # (batch, num_heads, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dim)
        if mask is not None:
            # Apply mask by setting -inf for masked positions
            attention_scores.masked_fill_(mask == 0, float('-inf'))

        # (batch, num_heads, seq_len, seq_len)-->
        # (batch, num_heads, seq_len, seq_len)
        attention_weights = attention_scores.softmax(dim=-1)  # Apply softmax
        if dropout is not None:
            attention_weights = dropout(attention_weights)
        # Return attention-weighted values and attention weights for visualization
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)-->
        # (batch, num_heads, seq_len, head_dim)
        return attention_weights @ value, attention_weights

    def forward(self, queries, keys, values, mask):
        # Project input embeddings to query, key, and value vectors
        queries = self.query_proj(queries)  # (batch, seq_len, model_dimension)
        keys = self.key_proj(keys)  # (batch, seq_len, model_dimension)
        values = self.value_proj(values)  # (batch, seq_len, model_dimension)

        # Reshape and split for multiple heads
        # (batch, seq_len, model_dimension)-->
        # (batch, seq_len, num_heads, head_dim)-->
        # (batch, num_heads, seq_len, head_dim)
        queries = queries.view(queries.shape[0], queries.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, seq_len, model_dimension)-->
        # (batch, seq_len, num_heads, head_dim)-->
        # (batch, num_heads, seq_len, head_dim)
        keys = keys.view(keys.shape[0], keys.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, seq_len, model_dimension)-->
        # (batch, seq_len, num_heads, head_dim)-->
        # (batch, num_heads, seq_len, head_dim)
        values = values.view(values.shape[0], values.shape[1], self.num_heads, self.head_dim).transpose(1, 2)

        # Calculate attention
        # (batch, num_heads, seq_len, head_dim), (batch, num_heads, seq_len, seq_len)
        attention_output, attention_weights = self.scaled_dot_product_attention(
            queries, keys, values, mask, self.dropout
        )

        # Concatenate heads and reshape
        # (batch, num_heads, seq_len, head_dim)-->
        # (batch, seq_len, num_heads, head_dim)-->
        # (batch, num_heads, seq_len, head_dim)-->
        # (batch, seq_len, model_dimension)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            attention_output.shape[0], -1, self.num_heads * self.head_dim
        )

        # Apply output projection
        # (batch, seq_len, model_dimension)-->
        # (batch, seq_len, model_dimension)
        output = self.output_proj(attention_output)

        return output
