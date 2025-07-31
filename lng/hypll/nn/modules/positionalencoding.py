# class HypPositionalEncoding(nn.Module):
#     """
#     Module that takes an image, splits it into patches and then embeds the patches on the hyperboloid.
#     """
#
#     def __init__(self, seq_manifold, ):
#         super().__init__()
#         self.lin = HLinear(in_features=, out_features=, manifold=manifold)
#         self.manifold = manifold
#
#     def forward(self, x: torch.Tensor) -> ManifoldTensor:
#         """
#         Takes an image of shape (B, C, H, W) and returns a tensor of shape (B, L, embed_dim). L is the number of patches.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor, shape (B, C, H, W).
#
#         Returns
#         -------
#         ManifoldTensor
#             Output tensor, shape (B, L, embed_dim).
#         """
#         p = self.lin(x)
#
#
#         return self
#         return self.manifold

# class HPositionalEncoding(nn.Module):
#     """
#     Module that takes an image, splits it into patches and then embeds the patches on the hyperboloid.
#     """
#
#     def __init__(self, seq_len: int, embed_dim, manifold):
#         super().__init__()
#         self.lin = HLinear(in_features=seq_len, out_features=embed_dim, manifold=manifold)
#         self.manifold = manifold
#
#     def forward(self, x: torch.Tensor) -> ManifoldTensor:
#         """
#         Takes an image of shape (B, C, H, W) and returns a tensor of shape (B, L, embed_dim). L is the number of patches.
#
#         Parameters
#         ----------
#         x : torch.Tensor
#             Input tensor, shape (B, C, H, W).
#
#         Returns
#         -------
#         ManifoldTensor
#             Output tensor, shape (B, L, embed_dim).
#         """
#         p = self.lin(x)
#         return self.manifold
