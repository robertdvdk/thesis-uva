"""
This module contains the PyTorch model definitions for a Hyperbolic
sequence-to-sequence Transformer, built as an analogue to the Euclidean version.
"""

import math
from typing import Optional

import torch
import torch.nn as nn

from hypll.manifolds.base import Manifold
from hypll.nn import HLinear, HMultiHeadAttention, HypLayerNorm, HypCLS, HypActivation, HypDropout
from hypll.tensors import ManifoldTensor


class HEmbedding(nn.Module):
    """
    Creates hyperbolic embeddings from token indices.

    This layer first uses a standard Euclidean embedding layer and then projects
    the resulting vectors onto the hyperboloid manifold by treating them as
    spatial coordinates and calculating the corresponding time coordinate.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> ManifoldTensor:
        """
        Takes token indices and returns points on the hyperboloid.

        Parameters
        ----------
        x : torch.Tensor
            A tensor of token indices, shape (batch_size, seq_len).

        Returns
        -------
        ManifoldTensor
            A tensor of points on the hyperboloid, shape (batch_size, seq_len, embedding_dim + 1).
        """
        # 1. Get standard Euclidean embeddings from the input token indices.
        # These vectors will serve as the SPATIAL coordinates of our hyperbolic points.
        # Shape: (batch_size, seq_len, embedding_dim)
        spatial_embeddings = self.embedding(x)

        # 2. Project these spatial coordinates onto the manifold.
        # The `project` method calculates the required time coordinate for each point
        # to ensure it lies on the hyperboloid, returning a valid ManifoldTensor.
        # Note: We wrap the Euclidean tensor so we can use the manifold's API.
        temp_tensor = ManifoldTensor(spatial_embeddings, manifold=self.manifold)
        hyperbolic_points = self.manifold.project(temp_tensor)
        return hyperbolic_points


class HPositionalEncoding(nn.Module):
    """
    Injects positional information into hyperbolic embeddings.

    This is done by mapping the input points to the tangent space at the origin
    (a Euclidean vector space), adding a learnable positional vector, and
    mapping the result back to the manifold.
    """

    def __init__(self, d_model: int, manifold: Manifold, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.manifold = manifold
        self.dropout = nn.Dropout(p=dropout)
        # Learnable positional vectors in the tangent space
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """
        Args:
            x (ManifoldTensor): Input tensor of shape [batch_size, seq_len, d_model+1].

        Returns:
            ManifoldTensor: Output tensor with positional information added.
        """
        seq_len = x.size(1)
        # Create position indices [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        # Get the corresponding learnable positional vectors
        positional_vectors = self.position_embedding(positions)


        # Map input points to tangent vectors at the origin
        tangent_vectors = self.manifold.logmap(x)
        tangent_vectors.tensor[..., 1:] = tangent_vectors.tensor[..., 1:] + positional_vectors
        tangent_vectors.tensor = self.dropout(tangent_vectors.tensor)
        # Map the new tangent vectors back to the manifold
        output = self.manifold.expmap(tangent_vectors)
        return output


class HTransformerEncoderLayer(nn.Module):
    """
    A single layer of the Hyperbolic Transformer encoder.
    Analogue of `nn.TransformerEncoderLayer`.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.self_attn = HMultiHeadAttention(d_model, nhead, manifold=manifold)
        self.linear1 = HLinear(d_model, dim_feedforward, manifold=manifold)
        self.dropout = HypDropout(dropout)
        self.linear2 = HLinear(dim_feedforward, d_model, manifold=manifold)
        self.norm1 = HypLayerNorm(manifold, d_model)
        self.norm2 = HypLayerNorm(manifold, d_model)

        self.relu = HypActivation(torch.nn.functional.relu, manifold=self.manifold)

    def forward(self, src: ManifoldTensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> ManifoldTensor:
        # Self-attention block
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        # Residual connection via Fréchet mean
        src = self.manifold.midpoint(src.stack(self.dropout(src2), dim=1), dim=1)
        src = self.norm1(src)

        # Feed-forward block
        ffn_output = self.linear2(self.dropout(self.relu(self.linear1(src))))

        # Residual connection via Fréchet mean
        src = self.manifold.midpoint(src.stack(self.dropout(ffn_output), dim=1), dim=1)
        src = self.norm2(src)
        return src


class HTransformerDecoderLayer(nn.Module):
    """
    A single layer of the Hyperbolic Transformer decoder.
    Analogue of `nn.TransformerDecoderLayer`.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.self_attn = HMultiHeadAttention(d_model, nhead, manifold=manifold)
        self.multihead_attn = HMultiHeadAttention(d_model, nhead, manifold=manifold)
        self.linear1 = HLinear(d_model, dim_feedforward, manifold=manifold)
        self.dropout = HypDropout(dropout)
        self.linear2 = HLinear(dim_feedforward, d_model, manifold=manifold)
        self.norm1 = HypLayerNorm(manifold, d_model)
        self.norm2 = HypLayerNorm(manifold, d_model)
        self.norm3 = HypLayerNorm(manifold, d_model)

        self.relu = HypActivation(torch.nn.functional.relu, manifold=self.manifold)

    def forward(self, tgt: ManifoldTensor, memory: ManifoldTensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> ManifoldTensor:
        # Masked self-attention on the target sequence
        print("tgt1", tgt.tensor.mean())
        tgt2 = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        print("tgt2", tgt2.tensor.mean())
        # FIXED: Apply dropout to the self-attention output.
        tgt = self.manifold.midpoint(tgt.stack(self.dropout(tgt2), dim=1), dim=1)
        print("tgt3", tgt.tensor.mean())
        tgt = self.norm1(tgt)
        print("tgt4", tgt.tensor.mean())

        # Cross-attention with the encoder's output memory
        tgt2 = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        print("tgt5", tgt2.tensor.mean())
        # FIXED: Apply dropout to the cross-attention output.
        tgt = self.manifold.midpoint(tgt.stack(self.dropout(tgt2), dim=1), dim=1)
        print("tgt6", tgt.tensor.mean())
        tgt = self.norm2(tgt)
        print("tgt7", tgt.tensor.mean())

        # Feed-forward block
        # FIXED: Apply dropout within the MLP.
        ffn_output = self.linear2(self.dropout(self.relu(self.linear1(tgt))))
        print("tgt8", ffn_output.tensor.mean())

        # FIXED: Apply dropout to the FFN output.
        tgt = self.manifold.midpoint(tgt.stack(self.dropout(ffn_output), dim=1), dim=1)
        print("tgt9", tgt.tensor.mean())
        tgt = self.norm3(tgt)
        print("tgt10", tgt.tensor.mean())
        return tgt

class HSeq2SeqTransformer(nn.Module):
    """
    A hyperbolic sequence-to-sequence Transformer model.
    """

    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 d_model: int, nhead: int, src_vocab_size: int,
                 tgt_vocab_size: int, dim_feedforward: int, dropout: float,
                 manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.d_model = d_model

        # --- Hyperbolic Components ---
        self.src_tok_emb = HEmbedding(src_vocab_size, d_model, manifold)
        self.tgt_tok_emb = HEmbedding(tgt_vocab_size, d_model, manifold)
        self.positional_encoding = HPositionalEncoding(d_model, manifold, dropout)
        self.encoder_layers = nn.ModuleList([
            HTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, manifold)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            HTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, manifold)
            for _ in range(num_decoder_layers)
        ])
        self.generator = nn.Linear(d_model + 1, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask for the decoder."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def encode(self, src: torch.Tensor, src_padding_mask: torch.Tensor) -> ManifoldTensor:
        """Encodes the source sequence into the hyperbolic manifold."""
        src_emb = self.src_tok_emb(src)

        src_emb = self.positional_encoding(src_emb)

        src_key_padding_mask = (src_padding_mask == 0)
        # Pass through encoder layers
        memory = src_emb

        for i, layer in enumerate(self.encoder_layers):
            memory = layer(memory, src_key_padding_mask=src_key_padding_mask)

        return memory

    def decode(self, tgt: torch.Tensor, memory: ManifoldTensor, tgt_mask: torch.Tensor,
               tgt_key_padding_mask: Optional[torch.Tensor] = None, memory_key_padding_mask: Optional[torch.Tensor] = None, prnt=False) -> ManifoldTensor:
        """Decodes the target sequence using the encoder's memory."""
        # Embed and add positional info
        tgt_emb = self.tgt_tok_emb(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)

        # Pass through decoder layers
        if tgt_key_padding_mask is not None:
            tgt_key_padding_mask = (tgt_key_padding_mask == 0)
        if memory_key_padding_mask is not None:
            memory_key_padding_mask = (memory_key_padding_mask == 0)
        output = tgt_emb
        for i, layer in enumerate(self.decoder_layers):
            output = layer(output, memory,
                           tgt_mask=tgt_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask)
        return output

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass for training."""
        # Create causal mask for the decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)

        # Run through encoder, decoder, and final generator
        memory = self.encode(src, src_padding_mask)
        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, src_padding_mask)
        ret = self.generator(outs.tensor)
        return ret