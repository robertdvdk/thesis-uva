from torch import nn

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor, TangentTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match
import torch
from .linear import HLinear

class HMultiHeadAttention(nn.Module):
    """
    A general multi-head attention module for hyperbolic spaces.

    This module supports two implementations:
    1.  'tangent': A hybrid approach that maps tensors to the tangent space,
        applies standard Euclidean attention, and maps the result back. This
        implementation is efficient as it leverages PyTorch's optimized MHA.

    2.  'naive': A fully hyperbolic attention mechanism as defined by Chen et al.
        This approach creates independent projection layers for each head to avoid
        invalid slicing of hyperbolic vectors.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        impl: str,
        bias: bool = True,
    ) -> None:
        super(HMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.impl = impl

        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})")
        # --------------------------------------------------------------------
        # Define layers based on the chosen implementation
        # --------------------------------------------------------------------
        if self.impl == "tangent":
            # For the tangent space, nn.MultiheadAttention handles all Q, K, V,
            # and Output projections internally.
            self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True)

        elif self.impl == "naive" or self.impl == "chen" or self.impl == "correction":
            # For the naive hyperbolic implementation, we must define our own
            # projection layers for each head to operate on the full manifold vectors.
            self.head_dim = embed_dim // num_heads

            # Create ModuleLists of HLinear layers, one projection per head.
            self.q_projs = nn.ModuleList(
                [HLinear(embed_dim, self.head_dim, manifold, impl, bias) for _ in range(num_heads)]
            )
            self.k_projs = nn.ModuleList(
                [HLinear(embed_dim, self.head_dim, manifold, impl, bias) for _ in range(num_heads)]
            )
            self.v_projs = nn.ModuleList(
                [HLinear(embed_dim, self.head_dim, manifold, impl, bias) for _ in range(num_heads)]
            )

            # A final projection layer to map the concatenated head outputs back to embed_dim.
            self.out_proj = HLinear(embed_dim, embed_dim, manifold, impl, bias)

        else:
            raise ValueError(f"Unknown implementation: {impl}")

    def forward(
        self,
        query: ManifoldTensor,
        key: ManifoldTensor,
        value: ManifoldTensor,
        key_padding_mask=None,
        attn_mask=None,
    ) -> ManifoldTensor:
        for t in [query, key, value]:
            check_if_manifolds_match(layer=self, input=t)
            check_if_man_dims_match(layer=self, man_dim=-1, input=t)

        if self.impl == "tangent":
            # 1. Map inputs to tangent space at the origin.
            query_tan = self.manifold.logmap(y=query)
            key_tan = self.manifold.logmap(y=key)
            value_tan = self.manifold.logmap(y=value)

            # 2. Perform standard Euclidean attention on the spatial components.
            # nn.MultiheadAttention handles all projections internally.
            attn_output_tan_spatial, _ = self.attn(
                query=query_tan.tensor[..., 1:],
                key=key_tan.tensor[..., 1:],
                value=value_tan.tensor[..., 1:],
                key_padding_mask=key_padding_mask,
                attn_mask=attn_mask,
            )

            # 3. Reconstruct the full tangent vector and map back to the manifold.
            time_component = torch.zeros_like(attn_output_tan_spatial[..., :1])
            attn_output_tan = torch.cat([time_component, attn_output_tan_spatial], dim=-1)
            output = self.manifold.expmap(
                TangentTensor(attn_output_tan, manifold=self.manifold, man_dim=-1)
            )
            return output

        elif self.impl == "naive" or self.impl == "chen" or self.impl == "correction":
            # 1. Compute attention for each head in parallel.
            head_outputs = []
            for i in range(self.num_heads):
                # Project the full input vectors for each head independently.
                q_i = self.q_projs[i](query)
                k_i = self.k_projs[i](key)
                v_i = self.v_projs[i](value)

                # Compute single-head attention on the manifold.
                output_i = self.manifold.multiheadattention(
                    query=q_i, key=k_i, value=v_i, att_module=None, impl=self.impl,
                    key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
                head_outputs.append(output_i)

            # 2. Concatenate head outputs using manifold-aware concatenation.
            # The `cat` method handles the proper combination of hyperbolic vectors.
            concatenated_heads = self.manifold.cat(head_outputs, dim=-1, impl=self.impl)

            # 3. Apply the final output projection.
            output = self.out_proj(concatenated_heads)
            return output
