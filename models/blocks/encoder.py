import torch
import torch.nn as nn
from components.blocks import MultiHeadAttentionBlock, ResidualBlock, FeedForwardBlock
from components.layers import LayerNormalizationLayer
from typing import List

class EncoderBlock(nn.Module):
    def __init__(
        self,
        multihead_attention_block: MultiHeadAttentionBlock,
        feed_forward_block: FeedForwardBlock,
        residual_blocks:List[ResidualBlock],
        layer_normalization_layers: List[LayerNormalizationLayer]
    ) -> None:

        super().__init__()
        self.multihead_attention_block = multihead_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_blocks = residual_blocks
        self.layer_normalization_layers = layer_normalization_layers

    def forward(self, x, src_mask):
        # 1. Apply the first residual block with self-attention
        x = self.residual_blocks[0](x, lambda x: self.multihead_attention_block(x, x, x, src_mask))
        # x shape: (batch_size, seq_len, features)
        
        x = self.layer_normalization_layers[0](x)

        # 2. Apply the second residual block with the feed-forward block
        x = self.residual_blocks[1](x, self.feed_forward_block)
        # x shape: (batch_size, seq_len, features)

        x = self.layer_normalization_layers[1](x)

        return x
