"""
This module defines the architectures for both a Hyperbolic Vision Transformer (HViT)
and a standard Euclidean Vision Transformer (ViT).

The Hyperbolic models use custom layers from the `hypll` library that perform
operations within a Hyperboloid manifold, while the Euclidean models use standard
`torch.nn` components.
"""
from typing import Any

from torch import nn
import torch

from hypll.manifolds.base import Manifold
from hypll.nn import HMultiHeadAttention, HLinear, HypLayerNorm, HypCLS, HPatchEmbedding
from hypll.nn import HypReLU


class HBlock(nn.Module):
    """A single block of the Hyperbolic Vision Transformer.

    This module implements the hyperbolic equivalent of a standard Transformer
    encoder block. It consists of multi-head self-attention and a feed-forward
    MLP, with hyperbolic layer normalization and residual connections.

    Attributes:
        manifold (Manifold): The manifold object defining the geometry.
        attn (HMultiHeadAttention): The hyperbolic multi-head self-attention layer.
        mlp (nn.Sequential): The feed-forward network (MLP) using hyperbolic layers.
        ln1 (HypLayerNorm): Layer normalization applied after the attention block.
        ln2 (HypLayerNorm): Layer normalization applied after the MLP block.
    """

    def __init__(
        self,
        manifold: Manifold,
        dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float,
    ):
        """Initializes the HBlock.

        Parameters
        ----------
        manifold : Manifold
            The hyperbolic manifold on which operations will be performed.
        dim : int
            The embedding dimension of the input and output.
        num_heads : int
            The number of attention heads.
        mlp_dim : int
            The hidden dimension of the MLP.
        dropout : float
            The dropout rate (currently unused but kept for API consistency).
        """
        super().__init__()
        self.manifold = manifold

        # Hyperbolic multi-head self-attention layer
        self.attn = HMultiHeadAttention(
            embed_dim=dim, num_heads=num_heads, manifold=manifold
        )

        # Hyperbolic feed-forward network
        self.mlp = nn.Sequential(
            HLinear(in_features=dim, out_features=mlp_dim, manifold=manifold),
            HypReLU(manifold=manifold),
            HLinear(in_features=mlp_dim, out_features=dim, manifold=manifold),
        )

        # Hyperbolic layer normalization
        self.ln1 = HypLayerNorm(manifold, dim)
        self.ln2 = HypLayerNorm(manifold, dim)

    def forward(self, x: "ManifoldTensor") -> "ManifoldTensor":
        """Defines the forward pass through the HBlock.

        Parameters
        ----------
        x : ManifoldTensor
            The input tensor on the manifold, shape (B, L, D+1).

        Returns
        -------
        ManifoldTensor
            The output tensor on the manifold, shape (B, L, D+1).
        """
        # First residual connection: input + attention_output
        z = self.attn(x, x, x)
        # In hyperbolic space, addition is replaced by the geometric mean (Fréchet mean).
        # We stack the input and output tensors along a new dimension (dim=1) and then
        # compute the mean along that dimension, effectively averaging the two points.
        x = self.manifold.midpoint(x=z.stack(x, dim=1), dim=1)
        x = self.ln1(x)

        # Second residual connection: input + mlp_output
        z = self.mlp(x)
        x = self.manifold.midpoint(x=z.stack(x, dim=1), dim=1)
        x = self.ln2(x)
        return x


class HViT(nn.Module):
    """A Vision Transformer implemented entirely in Hyperbolic space.

    This model adapts the standard ViT architecture by replacing all Euclidean
    components with their hyperbolic counterparts.

    Attributes:
        embed (HPatchEmbedding): The hyperbolic patch embedding layer.
        blocks (nn.ModuleList): A series of HBlock layers.
        classif (HypCLS): The final hyperbolic classification layer.
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
        num_layers: int,
        use_pos_enc: bool,
        manifold: Manifold,
    ):
        """Initializes the Hyperbolic Vision Transformer.

        Parameters
        ----------
        img_size : int
            The size (height and width) of the input images.
        patch_size : int
            The size of each square patch.
        in_channels : int
            The number of channels in the input images.
        embed_dim : int
            The dimension of the hyperbolic embeddings.
        num_heads : int
            The number of attention heads in each HBlock.
        num_classes : int
            The number of output classes for classification.
        num_layers : int
            The number of HBlock layers in the transformer.
        use_pos_enc : bool
            Whether to use positional encodings.
        manifold : Manifold
            The hyperbolic manifold object.
        """
        super().__init__()
        self.manifold = manifold

        # Project image patches into hyperbolic tangent space and then onto the manifold.
        self.embed = HPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_pos_enc=use_pos_enc,
            manifold=manifold,
        )

        # A stack of hyperbolic transformer blocks.
        self.blocks = nn.ModuleList(
            [
                HBlock(
                    manifold=manifold,
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=embed_dim,
                    dropout=0.0,
                )
                for _ in range(num_layers)
            ]
        )

        # Final classification layer that operates on the hyperboloid.
        self.classif = HypCLS(
            manifold=manifold, in_features=embed_dim, out_features=num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the full forward pass from image to classification logits.

        Parameters
        ----------
        x : torch.Tensor
            A batch of input images, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            The output classification logits.
        """
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)

        # Perform global average pooling via Fréchet mean over the sequence dimension (dim=1).
        # The squeeze call is removed as midpoint(..., keepdim=False) already removes the dimension.
        x = self.manifold.midpoint(x, dim=1)
        x = self.classif(x)
        return x


# ---------------------------------------------------------------------------- #
#                 Standard Euclidean Implementations for Comparison            #
# ---------------------------------------------------------------------------- #


class PatchEmbedding(nn.Module):
    """Standard Euclidean patch embedding layer.

    This module converts an image into a sequence of linearly projected,
    flattened patches, with optional learnable positional encodings.
    """

    def __init__(
        self, img_size: int, patch_size: int, in_channels: int, embed_dim: int, use_pos_enc: bool
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by the patch size.")

        self.patch_size = patch_size
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * (patch_size ** 2)

        # Layer to extract flattened patches from the image.
        self.patchify = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        # Linear projection layer to map flattened patches to the embedding dimension.
        self.proj = nn.Linear(patch_dim, embed_dim)

        # Optional learnable positional embeddings.
        self.use_pos_enc = use_pos_enc
        if self.use_pos_enc:
            self.pos_emb = nn.Parameter(torch.randn(1, num_patches, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, C, H, W)
        # Unfold into patches: (B, C*P*P, L), where L is num_patches.
        x = self.patchify(x)
        # Permute to get (B, L, C*P*P) for the linear layer.
        x = x.permute(0, 2, 1)
        # Project to (B, L, D).
        x = self.proj(x)
        if self.use_pos_enc:
            x = x + self.pos_emb
        return x


class ViTBlock(nn.Module):
    """Standard Euclidean Transformer encoder block."""

    def __init__(self, dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Attention block with residual connection
        attn_output, _ = self.attn(x, x, x)
        x = self.ln1(x + attn_output)

        # MLP block with residual connection
        mlp_output = self.mlp(x)
        x = self.ln2(x + mlp_output)
        return x


class ViT(nn.Module):
    """Standard Euclidean Vision Transformer model."""

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        in_channels: int,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
        num_layers: int,
        use_pos_enc: bool,
    ):
        super().__init__()

        self.embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_pos_enc=use_pos_enc,
        )
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=embed_dim * 4,  # Standard MLP expansion is 4x
                    dropout=0.1,
                )
                for _ in range(num_layers)
            ]
        )
        # A simple linear layer for the final classification.
        self.classif = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)

        # Standard global average pooling over the sequence length dimension.
        x = x.mean(dim=1)
        x = self.classif(x)
        return x