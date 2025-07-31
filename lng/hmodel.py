"""
This module contains the PyTorch model definitions for a Hyperbolic
sequence-to-sequence Transformer, built as an analogue to the Euclidean version.
"""

import torch.nn as nn
import torch
from hypll.manifolds.base import Manifold
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.nn import HLinear, HMultiHeadAttention, HypActivation, HypLayerNorm, HypDropout, HypResidual
from typing import Optional
import torch.nn.functional as F
import math

class LorentzMLR(nn.Module):
    """ Multinomial logistic regression (MLR) in the Lorentz model FROM BDEIR ET AL
    """

    def __init__(
            self,
            manifold: Manifold,
            num_features: int,
            num_classes: int
    ):
        super(LorentzMLR, self).__init__()

        self.manifold = manifold

        self.a = torch.nn.Parameter(torch.zeros(num_classes, ))
        self.z = torch.nn.Parameter(
            F.pad(torch.zeros(num_classes, num_features - 2), pad=(1, 0), value=1))  # z should not be (0,0)


        self.init_weights()

    def forward(self, x):
        # Hyperplane
        sqrt_mK = 1 / self.manifold.c().sqrt()
        norm_z = torch.norm(self.z, dim=-1)
        w_t = (torch.sinh(sqrt_mK * self.a) * norm_z)
        w_s = torch.cosh(sqrt_mK * self.a.view(-1, 1)) * self.z
        beta = torch.sqrt(-w_t ** 2 + torch.norm(w_s, dim=-1) ** 2)
        alpha = -w_t * x.narrow(-1, 0, 1) + (
                    torch.cosh(sqrt_mK * self.a) * torch.inner(x.narrow(-1, 1, x.shape[-1] - 1), self.z))

        d = self.manifold.c().sqrt() * torch.abs(torch.asinh(sqrt_mK * alpha / beta))  # Distance to hyperplane
        logits = torch.sign(alpha) * beta * d

        return logits

    def init_weights(self):
        stdv = 1. / math.sqrt(self.z.size(1))
        nn.init.uniform_(self.z, -stdv, stdv)
        nn.init.uniform_(self.a, -stdv, stdv)

class HEmbedding(nn.Module):
    """
    Creates hyperbolic embeddings from token indices.

    This layer first uses a standard Euclidean embedding layer and then projects
    the resulting vectors onto the hyperboloid manifold by treating them as
    spatial coordinates and calculating the corresponding time coordinate.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, manifold: Manifold, impl: str):
        super().__init__()
        self.manifold = manifold
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.impl = impl

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
        if self.impl == "tangent" or self.impl == "chen":
            # 1. Get standard Euclidean embeddings from the input token indices.
            # These vectors will serve as the SPATIAL coordinates of our hyperbolic points.
            # Shape: (batch_size, seq_len, embedding_dim)
            space = self.embedding(x)
            time = torch.zeros(space.shape[:-1], device=x.device).unsqueeze(-1)

            # 2. Project these spatial coordinates onto the manifold.
            # The `project` method calculates the required time coordinate for each point
            # to ensure it lies on the hyperboloid, returning a valid ManifoldTensor.
            # Note: We wrap the Euclidean tensor so we can use the manifold's API.
            tangent = TangentTensor(data=torch.cat([time, space], dim=-1), manifold=self.manifold, man_dim=-1)
            return self.manifold.expmap(tangent)
        elif self.impl == "naive" or self.impl == "correction":
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

    def __init__(self, d_model: int, manifold: Manifold, impl: str, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.manifold = manifold
        self.dropout = nn.Dropout(p=dropout)
        # Learnable positional vectors in the tangent space
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.impl = impl
        if impl == "chen":
            self.lin = HLinear(d_model, d_model, manifold=manifold, impl=impl)

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

        if self.impl == "tangent":
            # Map input points to tangent vectors at the origin
            tangent_vectors = self.manifold.logmap(y=x)
            tangent_vectors.tensor[..., 1:] = tangent_vectors.tensor[..., 1:] + positional_vectors
            tangent_vectors.tensor = self.dropout(tangent_vectors.tensor)
            # Map the new tangent vectors back to the manifold
            output = self.manifold.expmap(tangent_vectors)
            return output

        elif self.impl == "naive" or self.impl == "correction":
            space = self.dropout(x.tensor[..., 1:] + positional_vectors)
            time = torch.sqrt(torch.norm(space, p=2, dim=x.man_dim, keepdim=True) ** 2 + 1.0 / self.manifold.c())
            return ManifoldTensor(torch.cat([time, space], dim=x.man_dim), manifold=self.manifold, man_dim=x.man_dim)

        elif self.impl == "chen":
            return self.lin(x, bias=positional_vectors)




class TgtLayerNorm(nn.Module):
    """
    Applies Layer Normalization in the tangent space.

    Maps an input from the manifold to the tangent space at the origin,
    applies standard LayerNorm to the spatial components of the vector,
    and maps the result back to the manifold.
    """

    def __init__(self, manifold: Manifold, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.manifold = manifold
        # LayerNorm is applied to the spatial dimensions in the tangent space
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """
        Args:
            x (ManifoldTensor): Input tensor on the manifold.

        Returns:
            ManifoldTensor: Normalized tensor on the manifold.
        """
        # 1. Map input from the manifold to a Euclidean tangent space (at the origin)
        tangent_vec = self.manifold.logmap(y=x)

        # 2. Apply LayerNorm to the spatial components of the tangent vector.
        # The first component (dim -1, index 0) is time-like and is preserved.
        spatial_components = tangent_vec.tensor[..., 1:]
        time_component = tangent_vec.tensor[..., 0].unsqueeze(-1)

        normalized_spatial = self.norm(spatial_components)

        # 3. Recombine and update the tensor within the TangentTensor object
        tangent_vec.tensor = torch.cat([time_component, normalized_spatial], dim=-1)

        # 4. Map the normalized tangent vector back to the manifold
        output = self.manifold.expmap(tangent_vec)
        return output


class TgtActivation(nn.Module):
    """
    Applies a non-linear activation function in the tangent space.
    """

    def __init__(self, activation_fn, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.activation = activation_fn

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """
        Args:
            x (ManifoldTensor): Input tensor on the manifold.

        Returns:
            ManifoldTensor: Output tensor after applying activation.
        """
        # 1. Map to tangent space
        tangent_vec = self.manifold.logmap(y=x)

        # 2. Apply activation to the spatial components
        spatial_components = tangent_vec.tensor[..., 1:]
        time_component = tangent_vec.tensor[..., 0].unsqueeze(-1)

        activated_spatial = self.activation(spatial_components)

        # 3. Recombine and update the tensor
        tangent_vec.tensor = torch.cat([time_component, activated_spatial], dim=-1)

        # 4. Map back to the manifold
        output = self.manifold.expmap(tangent_vec)
        return output


class TgtDropout(nn.Module):
    """
    Applies Dropout in the tangent space.
    """

    def __init__(self, p: float, manifold: Manifold):
        super().__init__()
        self.manifold = manifold
        self.dropout = nn.Dropout(p=p)

    def forward(self, x: ManifoldTensor) -> ManifoldTensor:
        """
        Args:
            x (ManifoldTensor): Input tensor on the manifold.

        Returns:
            ManifoldTensor: Output tensor with dropout applied.
        """
        # nn.Dropout automatically handles self.training state
        # 1. Map to tangent space
        tangent_vec = self.manifold.logmap(y=x)

        # 2. Apply dropout to the entire tangent vector representation.
        # Since the time coordinate is zero at the origin's tangent space,
        # dropout effectively operates only on the spatial components.
        tangent_vec.tensor = self.dropout(tangent_vec.tensor)

        # 3. Map back to the manifold
        output = self.manifold.expmap(tangent_vec)
        return output

class TgtResidual(nn.Module):
    def __init__(self, manifold: Manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: ManifoldTensor, y: ManifoldTensor) -> ManifoldTensor:
        tangent_1 = self.manifold.logmap(y=x)
        tangent_2 = self.manifold.logmap(y=y)
        tangent_1.tensor += tangent_2.tensor
        return self.manifold.expmap(tangent_1)

class HTransformerEncoderLayer(nn.Module):
    """
    A single layer of the Hyperbolic Transformer encoder.
    Analogue of `nn.TransformerEncoderLayer`.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, manifold: Manifold, impl: str):
        super().__init__()
        self.manifold = manifold
        self.self_attn = HMultiHeadAttention(d_model, nhead, manifold=manifold, impl=impl)
        self.linear1 = HLinear(d_model, dim_feedforward, manifold=manifold, impl=impl)
        self.linear2 = HLinear(dim_feedforward, d_model, manifold=manifold, impl=impl)
        if impl == "tangent":
            self.dropout = TgtDropout(dropout, manifold=manifold)
            self.norm1 = TgtLayerNorm(manifold, d_model)
            self.norm2 = TgtLayerNorm(manifold, d_model)
            self.res = TgtResidual(manifold=manifold)
            self.relu = TgtActivation(torch.nn.functional.relu, manifold=self.manifold)
        elif impl == "naive" or impl == "correction":
            self.dropout = HypDropout(dropout)
            self.norm1 = HypLayerNorm(manifold, d_model)
            self.norm2 = HypLayerNorm(manifold, d_model)
            self.res = HypResidual()
            self.relu = HypActivation(torch.nn.functional.relu, manifold=self.manifold)


    def forward(self, src: ManifoldTensor, src_key_padding_mask: Optional[torch.Tensor] = None) -> ManifoldTensor:
        # Self-attention block
        src2 = self.self_attn(src, src, src, key_padding_mask=src_key_padding_mask)
        # Residual connection via Fréchet mean
        src = self.res(src, self.dropout(src2))
        src = self.norm1(src)

        # Feed-forward block
        ffn_output = self.linear2(self.dropout(self.relu(self.linear1(src))))

        # Residual connection via Fréchet mean
        src = self.res(src, self.dropout(ffn_output))
        src = self.norm2(src)
        return src

class HTransformerDecoderLayer(nn.Module):
    """
    A single layer of the Hyperbolic Transformer decoder.
    Analogue of `nn.TransformerDecoderLayer`.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float, manifold: Manifold, impl: str):
        super().__init__()
        self.manifold = manifold
        self.self_attn = HMultiHeadAttention(d_model, nhead, manifold=manifold, impl=impl)
        self.multihead_attn = HMultiHeadAttention(d_model, nhead, manifold=manifold, impl=impl)
        self.linear1 = HLinear(d_model, dim_feedforward, manifold=manifold, impl=impl)
        self.linear2 = HLinear(dim_feedforward, d_model, manifold=manifold, impl=impl)
        if impl == "tangent":
            self.dropout = TgtDropout(dropout, manifold=manifold)
            self.norm1 = TgtLayerNorm(manifold, d_model)
            self.norm2 = TgtLayerNorm(manifold, d_model)
            self.norm3 = TgtLayerNorm(manifold, d_model)
            self.res = TgtResidual(manifold=manifold)
            self.relu = TgtActivation(torch.nn.functional.relu, manifold=self.manifold)
        elif impl == "naive" or impl == "correction":
            self.dropout = HypDropout(dropout)
            self.norm1 = HypLayerNorm(manifold, d_model)
            self.norm2 = HypLayerNorm(manifold, d_model)
            self.norm3 = HypLayerNorm(manifold, d_model)
            self.res = HypResidual()
            self.relu = HypActivation(torch.nn.functional.relu, manifold=self.manifold)

    def forward(self, tgt: ManifoldTensor, memory: ManifoldTensor,
                tgt_mask: Optional[torch.Tensor] = None,
                tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None) -> ManifoldTensor:
        # Masked self-attention on the target sequence
        tgt2 = self.self_attn(tgt, tgt, tgt, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)
        # FIXED: Apply dropout to the self-attention output.
        tgt = self.res(tgt, self.dropout(tgt2))
        tgt = self.norm1(tgt)

        # Cross-attention with the encoder's output memory
        tgt2 = self.multihead_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask)
        # FIXED: Apply dropout to the cross-attention output.
        tgt = self.res(tgt, self.dropout(tgt2))
        tgt = self.norm2(tgt)

        # Feed-forward block
        # FIXED: Apply dropout within the MLP.
        ffn_output = self.linear2(self.dropout(self.relu(self.linear1(tgt))))

        # FIXED: Apply dropout to the FFN output.
        tgt = self.res(tgt, self.dropout(ffn_output))
        tgt = self.norm3(tgt)
        return tgt

# @torch.compile
class HSeq2SeqTransformer(nn.Module):
    """
    A hyperbolic sequence-to-sequence Transformer model.
    """

    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 d_model: int, nhead: int, src_vocab_size: int,
                 tgt_vocab_size: int, dim_feedforward: int, dropout: float,
                 manifold: Manifold, impl: str):
        super().__init__()
        self.manifold = manifold
        self.src_tok_emb = HEmbedding(src_vocab_size, d_model, manifold, impl=impl)
        self.tgt_tok_emb = HEmbedding(tgt_vocab_size, d_model, manifold, impl=impl)
        self.positional_encoding = HPositionalEncoding(d_model, manifold, impl=impl, dropout=dropout)

        self.encoder_layers = nn.ModuleList([
            HTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, manifold, impl=impl)
            for _ in range(num_encoder_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            HTransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, manifold, impl=impl)
            for _ in range(num_decoder_layers)
        ])
        # if gen_impl == "LorentzMLR":
        #     self.generator = LorentzMLR(manifold=self.manifold, num_features=d_model + 1, num_classes=tgt_vocab_size)
        # elif gen_impl == "HLinear":
        #     self.generator = HLinear(d_model, tgt_vocab_size, manifold, impl=impl)
        # elif gen_impl == "Linear":
        self.generator = nn.Linear(d_model + 1, tgt_vocab_size)
        # else:
        #     raise NotImplementedError


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
        # if self.gen_impl == "HLinear":
        #     ret = self.generator(outs)
        #     return ret.tensor[..., 1:]
        # else:
        ret = self.generator(outs.tensor)
        return ret