from torch import nn
import torch

from hypll.nn import HMultiHeadAttention, HLinear, HypLayerNorm, HypCLS, HPatchEmbedding
from hypll.nn import HypReLU


class HBlock(nn.Module):
    def __init__(self, manifold, dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.manifold = manifold
        self.attn = HMultiHeadAttention(dim, num_heads, manifold=manifold)
        self.mlp = nn.Sequential(
            HLinear(dim, mlp_dim, manifold=manifold),
            HypReLU(manifold=manifold),
            HLinear(mlp_dim, dim, manifold=manifold),
        )
        self.ln1 = HypLayerNorm(manifold, dim)
        self.ln2 = HypLayerNorm(manifold, dim)

    def forward(self, x):
        z = self.attn(x, x, x)
        x = self.manifold.midpoint(z.stack(x, dim=-2)).squeeze()
        x = self.ln1(x)
        z = self.mlp(x)
        x = self.manifold.midpoint(z.stack(x, dim=-2)).squeeze()
        x = self.ln2(x)
        return x


class HViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        num_heads,
        num_classes,
        num_layers,
        use_pos_enc,
        manifold,
    ):
        super().__init__()

        self.embed = HPatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_pos_enc=use_pos_enc,
            manifold=manifold,
        )
        self.manifold = manifold
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                HBlock(manifold, embed_dim, num_heads, embed_dim, dropout=0)
            )
        self.classif = HypCLS(manifold, embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for idx, block in enumerate(self.blocks):
            x = block(x)
        x = self.manifold.midpoint(x).squeeze()
        x = self.classif(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, use_pos_enc):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.patchify = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.proj = nn.Linear(self.in_channels * self.patch_size**2, self.embed_dim)

        num_patches = (self.img_size // self.patch_size) ** 2
        self.use_pos_enc = use_pos_enc
        if self.use_pos_enc:
            self.pos_emb = nn.Parameter(torch.randn(num_patches, self.embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patchify(x).permute(0, 2, 1)
        x = self.proj(x)
        if self.use_pos_enc:
            x += self.pos_emb
        return x


class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        z, _ = self.attn(x, x, x)
        x = x + z
        x = self.ln1(x)

        z = self.mlp(x)
        x = x + z
        x = self.ln2(x)
        return x


class ViT(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        in_channels,
        embed_dim,
        num_heads,
        num_classes,
        num_layers,
        use_pos_enc,
    ):
        super().__init__()

        self.embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            use_pos_enc=use_pos_enc,
        )
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(ViTBlock(embed_dim, num_heads, embed_dim, dropout=0))
        self.classif = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)
        x = self.classif(x)
        return x
