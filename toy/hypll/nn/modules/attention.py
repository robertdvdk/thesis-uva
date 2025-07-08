from torch import nn

from hypll.manifolds import Manifold
from hypll.tensors import ManifoldTensor
from hypll.utils.layer_utils import check_if_man_dims_match, check_if_manifolds_match

from .linear import HLinear


class HMultiHeadAttention(nn.Module):
    """
    Lorentzian multi-head attention layer, as defined by Chen et al.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        manifold: Manifold,
        bias: bool = True,
    ) -> None:
        super(HMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.manifold = manifold
        self.has_bias = bias

        self.W_Q = HLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            manifold=manifold,
            bias=bias,
            num_heads=num_heads,
        )
        self.W_K = HLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            manifold=manifold,
            bias=bias,
            num_heads=num_heads,
        )
        self.W_V = HLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            manifold=manifold,
            bias=bias,
            num_heads=num_heads,
        )
        self.W_O = HLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            manifold=manifold,
            bias=bias,
            num_heads=0,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.W_Q.reset_parameters()
        self.W_K.reset_parameters()
        self.W_V.reset_parameters()
        self.W_O.reset_parameters()

    def forward(
        self, query: ManifoldTensor, key: ManifoldTensor, value: ManifoldTensor, key_padding_mask = None, attn_mask = None
    ) -> ManifoldTensor:
        """
        Given query, key, and value tensors, compute the multi-head attention in the manifold on which the tensors lie.

        Parameters
        ----------
        query : ManifoldTensor
            Query tensor, (B, L, D).
        key : ManifoldTensor
            Key tensor, (B, L, D).
        value : ManifoldTensor
            Value tensor, (B, L, D).

        Returns
        -------
        ManifoldTensor
            Multi-head attention output tensor, (B, L, D).
        """
        check_if_manifolds_match(layer=self, input=query)
        check_if_manifolds_match(layer=self, input=key)
        check_if_manifolds_match(layer=self, input=value)
        check_if_man_dims_match(layer=self, man_dim=-1, input=query)
        check_if_man_dims_match(layer=self, man_dim=-1, input=key)
        check_if_man_dims_match(layer=self, man_dim=-1, input=value)
        return self.manifold.multiheadattention(
            query=query,
            key=key,
            value=value,
            W_Q=self.W_Q,
            W_K=self.W_K,
            W_V=self.W_V,
            W_O=self.W_O,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask
        )
