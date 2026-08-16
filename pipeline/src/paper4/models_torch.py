from __future__ import annotations

import math


def require_torch():
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PyTorch is required for xLSTM and Transformer sequence models. "
            "Install torch or run inside the configured AutoResearch environment."
        ) from exc
    return torch, nn


class TorchModelFactory:
    @staticmethod
    def xlstm(n_dynamic: int, n_static: int, cfg: dict):
        torch, nn = require_torch()

        class ScalarXLSTMCell(nn.Module):
            def __init__(self, input_size, hidden_size):
                super().__init__()
                self.hidden_size = hidden_size
                self.proj = nn.Linear(input_size + hidden_size, 4 * hidden_size)

            def forward(self, x, state):
                h, c, n, m = state
                xh = torch.cat((x.contiguous(), h.contiguous()), dim=-1)
                z, log_i, log_f, o = self.proj(xh).chunk(4, dim=-1)
                z = torch.tanh(z)
                o = torch.sigmoid(o)
                log_i = log_i.clamp(-12, 12)
                log_f = log_f.clamp(-12, 12)
                # `torch.maximum` backpropagates through `where` on MPS and can
                # abort in Apple's MPSGraph runtime. `logaddexp` gives the same
                # stable log-domain normalization role without that kernel.
                m_new = torch.logaddexp(log_f + m, log_i)
                i = torch.exp(log_i - m_new)
                f = torch.exp(log_f + m - m_new)
                c_new = f * c + i * z
                n_new = f * n + i
                h_new = o * c_new / (n_new + 1e-6)
                return h_new, (h_new, c_new, n_new, m_new)

        class XLSTMRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = int(cfg["sequence_models"]["hidden_size"])
                dropout = float(cfg["sequence_models"].get("dropout", 0.0))
                self.static_encoder = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU(), nn.Dropout(dropout))
                self.input_encoder = nn.Linear(n_dynamic + hidden, hidden)
                self.cell = ScalarXLSTMCell(hidden, hidden)
                self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

            def forward(self, x_dyn, x_static):
                x_dyn = x_dyn.contiguous()
                s = self.static_encoder(x_static.contiguous())
                s_rep = s[:, None, :].expand(-1, x_dyn.size(1), -1).contiguous()
                x = torch.relu(self.input_encoder(torch.cat((x_dyn, s_rep), dim=-1)))
                h = torch.zeros(x.size(0), s.size(-1), device=x.device)
                c = torch.zeros_like(h)
                n = torch.zeros_like(h)
                m = torch.zeros_like(h)
                for t in range(x.size(1)):
                    h, (h, c, n, m) = self.cell(x[:, t, :], (h, c, n, m))
                return self.head(h).squeeze(-1)

        return XLSTMRegressor()

    @staticmethod
    def transformer(n_dynamic: int, n_static: int, cfg: dict):
        torch, nn = require_torch()

        class PositionalEncoding(nn.Module):
            def __init__(self, d_model: int, max_len: int = 1000):
                super().__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                self.register_buffer("pe", pe.unsqueeze(0))

            def forward(self, x):
                return x + self.pe[:, : x.size(1), :]

        class TransformerRegressor(nn.Module):
            def __init__(self):
                super().__init__()
                hidden = int(cfg["sequence_models"]["hidden_size"])
                layers = int(cfg["sequence_models"].get("num_layers", 1))
                dropout = float(cfg["sequence_models"].get("dropout", 0.0))
                heads = int(cfg["sequence_models"].get("attention_heads", 4))
                self.static_encoder = nn.Sequential(nn.Linear(n_static, hidden), nn.ReLU())
                self.input_proj = nn.Linear(n_dynamic + hidden, hidden)
                self.pos = PositionalEncoding(hidden, max_len=int(cfg["sequence_models"]["sequence_length"]) + 5)
                self.encoder = nn.ModuleList([SelfAttentionBlock(hidden, heads, dropout) for _ in range(layers)])
                self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))

            def forward(self, x_dyn, x_static):
                x_dyn = x_dyn.contiguous()
                s = self.static_encoder(x_static.contiguous())
                s_rep = s[:, None, :].expand(-1, x_dyn.size(1), -1).contiguous()
                x = self.input_proj(torch.cat((x_dyn, s_rep), dim=-1))
                x = self.pos(x)

                for block in self.encoder:
                    x = block(x)
                return self.head(x[:, -1, :]).squeeze(-1)

        class SelfAttentionBlock(nn.Module):
            def __init__(self, hidden: int, heads: int, dropout: float):
                super().__init__()
                if hidden % heads != 0:
                    raise ValueError(f"hidden_size={hidden} must be divisible by attention_heads={heads}")
                self.heads = heads
                self.head_dim = hidden // heads
                self.norm_attn = nn.LayerNorm(hidden)
                self.qkv = nn.Linear(hidden, hidden * 3)
                self.out_proj = nn.Linear(hidden, hidden)
                self.drop = nn.Dropout(dropout)
                self.norm_ff = nn.LayerNorm(hidden)
                self.ff = nn.Sequential(
                    nn.Linear(hidden, hidden * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden * 4, hidden),
                    nn.Dropout(dropout),
                )

            def forward(self, x):
                bsz, seq_len, hidden = x.shape
                residual = x
                q, k, v = self.qkv(self.norm_attn(x)).chunk(3, dim=-1)
                q = q.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
                k = k.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)
                v = v.view(bsz, seq_len, self.heads, self.head_dim).transpose(1, 2)

                scores = q @ k.transpose(-2, -1)
                scores = scores / math.sqrt(self.head_dim)
                attn = torch.softmax(scores, dim=-1)
                y = attn @ v
                y = y.transpose(1, 2).contiguous().view(bsz, seq_len, hidden)
                x = residual + self.drop(self.out_proj(y))
                return x + self.ff(self.norm_ff(x))

        return TransformerRegressor()
