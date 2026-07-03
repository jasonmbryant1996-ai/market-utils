"""
model_arch.py
RegimeTransformer — identical to the architecture used during training.
Must match exactly: same d_model, nhead, num_layers, dim_ffn, dropout.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(1))   # (max_len, 1, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[: x.size(0)])


class RegimeTransformer(nn.Module):
    """
    Transformer-based regime classifier.

    Input : (batch, lookback, input_dim)
    Output: (batch, num_classes)  — raw logits
    """

    def __init__(
        self,
        input_dim:  int = 35,
        d_model:    int = 128,
        nhead:      int = 4,
        num_layers: int = 3,
        dim_ffn:    int = 512,
        dropout:    float = 0.2,
        num_classes: int = 3,
    ):
        super().__init__()
        self.d_model    = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model        = d_model,
            nhead          = nhead,
            dim_feedforward= dim_ffn,
            dropout        = dropout,
            batch_first    = False,
            norm_first     = True,          # Pre-LN
        )
        self.transformer = nn.TransformerEncoder(
            enc_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        x = src.transpose(0, 1)                           # (seq, batch, feat)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = self.norm(x[-1])                              # last token
        return self.head(x)
