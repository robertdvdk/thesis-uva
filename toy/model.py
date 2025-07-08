"""
Contains the PyTorch model definitions for the Transformer.
Includes the PositionalEncoding module and the main Seq2SeqTransformer class.
"""

import math
import torch
import torch.nn as nn
from torch.nn import Transformer


class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    This version uses a learnable nn.Embedding layer.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.position_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            torch.Tensor: Output tensor with positional information added.
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        positional_embeddings = self.position_embedding(positions)
        x = x + positional_embeddings
        return self.dropout(x)


class Seq2SeqTransformer(nn.Module):
    """
    A sequence-to-sequence Transformer model that is structured for both
    training and inference, with separate encode and decode methods.
    """
    def __init__(self, num_encoder_layers: int, num_decoder_layers: int,
                 d_model: int, nhead: int, src_vocab_size: int,
                 tgt_vocab_size: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.transformer = Transformer(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)
        self.src_tok_emb = nn.Embedding(src_vocab_size, d_model)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.d_model = d_model

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generates a causal mask for the decoder."""
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor, prnt=False) -> torch.Tensor:
        """
        Encodes the source sequence.
        Args:
            src (torch.Tensor): Source sequence tensor, shape [batch_size, src_len].
            src_padding_mask (torch.Tensor): Mask for source padding, shape [batch_size, src_len].
        Returns:
            torch.Tensor: The encoder's output memory.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.d_model))
        # The nn.Transformer.encoder expects padding masks where True indicates a padded token. Invert it.
        src_key_padding_mask = (src_key_padding_mask == 0)
        return self.transformer.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor,
               tgt_padding_mask: torch.Tensor = None, memory_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Decodes the target sequence using the encoder's memory.
        Args:
            tgt (torch.Tensor): The target sequence so far.
            memory (torch.Tensor): The encoder's output memory.
            tgt_mask (torch.Tensor): The causal mask for the target sequence.
            tgt_padding_mask (torch.Tensor): The mask for target padding.
            memory_key_padding_mask (torch.Tensor): The mask for the encoder memory from the source.
        Returns:
            torch.Tensor: Raw output from the decoder.
        """
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.d_model))

        if tgt_padding_mask is not None:
            tgt_padding_mask = (tgt_padding_mask == 0)

        if memory_key_padding_mask is not None:
            memory_key_padding_mask = (memory_key_padding_mask == 0)

        return self.transformer.decoder(tgt_emb, memory,
                                        tgt_mask=tgt_mask,
                                        tgt_key_padding_mask=tgt_padding_mask,
                                        memory_key_padding_mask=memory_key_padding_mask)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_padding_mask: torch.Tensor, tgt_padding_mask: torch.Tensor, prnt=False) -> torch.Tensor:
        """
        Forward pass for training.
        """
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(1), tgt.device)
        memory = self.encode(src, src_padding_mask, prnt)
        outs = self.decode(tgt, memory, tgt_mask, tgt_padding_mask, src_padding_mask)
        return self.generator(outs)