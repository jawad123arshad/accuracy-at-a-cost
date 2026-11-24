"""
Three small, from-scratch sequence encoders (BiGRU / Transformer / a minimal
Mamba-style selective SSM), each producing a pooled representation of size
config.HIDDEN_DIM, plus a shared MultiTaskModel wrapper with per-task heads.

Notes:
- The Mamba block here is a compact, pure-PyTorch reimplementation of the
  selective-SSM recurrence (not the official CUDA `mamba_ssm` package, which
  is difficult to build on Windows/consumer GPUs). It follows the mechanism in
  Gu & Dao (2023) but is far slower than the fused-kernel package.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


# ---------------------------------------------------------------------------
# Shared building block
# ---------------------------------------------------------------------------
def masked_mean_pool(hidden, lengths):
    """hidden: (B, T, H); lengths: (B,) -> (B, H)"""
    B, T, H = hidden.shape
    mask = (torch.arange(T, device=hidden.device)[None, :] < lengths[:, None]).float()
    mask = mask.unsqueeze(-1)  # (B, T, 1)
    summed = (hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


# ---------------------------------------------------------------------------
# 1. BiGRU encoder
# ---------------------------------------------------------------------------
class BiGRUEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, C.EMBED_DIM, padding_idx=0)
        self.gru = nn.GRU(
            C.EMBED_DIM, C.HIDDEN_DIM // 2, num_layers=C.NUM_LAYERS,
            batch_first=True, bidirectional=True, dropout=C.DROPOUT if C.NUM_LAYERS > 1 else 0.0,
        )
        self.out_dim = C.HIDDEN_DIM

    def forward(self, input_ids, lengths):
        x = self.embed(input_ids)
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=input_ids.size(1))
        return masked_mean_pool(out, lengths)


# ---------------------------------------------------------------------------
# 2. Small Transformer encoder
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=C.MAX_SEQ_LEN + 4):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoderSmall(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, C.HIDDEN_DIM, padding_idx=0)
        self.pos = PositionalEncoding(C.HIDDEN_DIM)
        layer = nn.TransformerEncoderLayer(
            d_model=C.HIDDEN_DIM, nhead=C.NUM_HEADS, dim_feedforward=C.HIDDEN_DIM * 2,
            dropout=C.DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=C.NUM_LAYERS)
        self.out_dim = C.HIDDEN_DIM

    def forward(self, input_ids, lengths):
        T = input_ids.size(1)
        pad_mask = torch.arange(T, device=input_ids.device)[None, :] >= lengths[:, None]  # True = pad
        x = self.embed(input_ids) * math.sqrt(C.HIDDEN_DIM)
        x = self.pos(x)
        out = self.encoder(x, src_key_padding_mask=pad_mask)
        return masked_mean_pool(out, lengths)


# ---------------------------------------------------------------------------
# 3. Minimal selective-SSM ("Mamba-style") block, pure PyTorch
# ---------------------------------------------------------------------------
class SelectiveSSMBlock(nn.Module):
    """A compact reimplementation of the selective-scan recurrence:
        h_t = A_bar_t * h_{t-1} + B_bar_t * x_t ;  y_t = C_t . h_t
    with input-dependent (selective) discretized A, B, C, following the
    mechanism described in Gu & Dao (2023). State size and expansion factor
    are kept small on purpose for 6GB-VRAM training.
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=3, padding=2, groups=self.d_inner)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)  # -> B, C, dt (per step, selective)
        self.dt_proj = nn.Linear(1, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1)  # each (B, T, d_inner)

        x_conv = self.conv1d(x_in.transpose(1, 2))[:, :, :T].transpose(1, 2)  # causal conv
        x_conv = F.silu(x_conv)

        proj = self.x_proj(x_conv)  # (B, T, 2*d_state + 1)
        B_sel, C_sel, dt_raw = torch.split(proj, [self.d_state, self.d_state, 1], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))  # (B, T, d_inner)

        A = -torch.exp(self.A_log)  # (d_inner, d_state), stays negative -> stable
        # Discretize (zero-order hold approximation) and scan sequentially.
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        ys = []
        for t in range(T):
            dt_t = dt[:, t, :].unsqueeze(-1)                    # (B, d_inner, 1)
            A_bar = torch.exp(dt_t * A.unsqueeze(0))            # (B, d_inner, d_state)
            B_bar = dt_t * B_sel[:, t, :].unsqueeze(1)          # (B, d_inner, d_state)
            h = A_bar * h + B_bar * x_conv[:, t, :].unsqueeze(-1)
            y_t = (h * C_sel[:, t, :].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, d_inner)
        y = y + x_conv * self.D
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaEncoderSmall(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, C.HIDDEN_DIM, padding_idx=0)
        self.blocks = nn.ModuleList([SelectiveSSMBlock(C.HIDDEN_DIM) for _ in range(C.NUM_LAYERS)])
        self.norms = nn.ModuleList([nn.LayerNorm(C.HIDDEN_DIM) for _ in range(C.NUM_LAYERS)])
        self.dropout = nn.Dropout(C.DROPOUT)
        self.out_dim = C.HIDDEN_DIM

    def forward(self, input_ids, lengths):
        x = self.embed(input_ids)
        for block, norm in zip(self.blocks, self.norms):
            x = x + self.dropout(block(norm(x)))
        return masked_mean_pool(x, lengths)


ENCODERS = {
    "bigru": BiGRUEncoder,
    "transformer": TransformerEncoderSmall,
    "mamba": MambaEncoderSmall,
}


# ---------------------------------------------------------------------------
# Multitask wrapper
# ---------------------------------------------------------------------------
class MultiTaskModel(nn.Module):
    def __init__(self, arch, vocab_size, num_classes, active_tasks):
        """active_tasks: subset of {"classification", "verdict"} -- a
        single-task run passes just one of them."""
        super().__init__()
        assert arch in ENCODERS, f"unknown arch {arch}"
        self.arch = arch
        self.active_tasks = list(active_tasks)
        self.encoder = ENCODERS[arch](vocab_size)
        heads = {}
        if "classification" in self.active_tasks:
            heads["classification"] = nn.Linear(self.encoder.out_dim, num_classes)
        if "verdict" in self.active_tasks:
            heads["verdict"] = nn.Linear(self.encoder.out_dim, 2)
        self.heads = nn.ModuleDict(heads)

    def forward(self, input_ids, lengths):
        pooled = self.encoder(input_ids, lengths)
        return {task: head(pooled) for task, head in self.heads.items()}
