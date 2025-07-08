import torch
import torch.nn as nn
from torch.nn import Module
from hypll.tensors import ManifoldTensor


class HPatchEmbedding(Module):
    """Hyperbolic fully connected linear layer"""

    def __init__(
        self, img_size, patch_size, in_channels, embed_dim, use_pos_enc, manifold
    ) -> None:
        super(HPatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.manifold = manifold
        # TODO: torch stores weights transposed supposedly due to efficiency
        # https://discuss.pytorch.org/t/why-does-the-linear-module-seems-to-do-unnecessary-transposing/6277/7
        # We may want to do the same
        self.z, _ = self.manifold.construct_dl_parameters(
            in_features=in_channels * patch_size**2, out_features=embed_dim, bias=False
        )
        if use_pos_enc:
            self.pos_enc = nn.Parameter(torch.randn(self.num_patches, self.embed_dim))
        else:
            self.pos_enc = torch.zeros(self.num_patches, self.embed_dim)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        self.manifold.reset_parameters(self.z, bias=None)

    def forward(self, x: torch.Tensor) -> ManifoldTensor:
        self.pos_enc = self.pos_enc.to(device=x.device)
        return self.manifold.patch_embedding(
            x=x, z=self.z, positional_encoding=self.pos_enc, patch_size=self.patch_size
        )