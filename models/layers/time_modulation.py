"""AdaLN-style target-time modulation for the renderer (per-sublayer shift/scale/gate, init = identity)."""

import torch
import torch.nn as nn


class TimeModulation(nn.Module):
    """Map target-time cond to per-sublayer (shift, scale, gate); output [N, n_sublayers, 3, hidden_dim]."""

    def __init__(self, cond_dim: int, hidden_dim: int, n_sublayers: int):
        super().__init__()
        self.n_sublayers = n_sublayers
        self.hidden_dim = hidden_dim
        self.act = nn.SiLU()
        self.proj = nn.Linear(cond_dim, n_sublayers * 3 * hidden_dim)
        self.reset_parameters()

    def reset_parameters(self):
        """Zero weight + bias (shift=0, scale=0, gate=1)."""
        nn.init.zeros_(self.proj.weight)
        bias = torch.zeros(self.n_sublayers, 3, self.hidden_dim)
        bias[:, 2, :] = 1.0  # gate bias = 1 → sublayer fully applied at init
        with torch.no_grad():
            self.proj.bias.copy_(bias.reshape(-1))

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        # cond: [N, cond_dim] → [N, n_sublayers, 3, hidden_dim]
        m = self.proj(self.act(cond))
        return m.reshape(cond.shape[0], self.n_sublayers, 3, self.hidden_dim)


def modulate(x_norm: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """AdaLN: (1 + scale) * norm(x) + shift. shift/scale broadcast over tokens."""
    return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
