"""
Small fixed circuit inspired by MaleCNS connectivity patterns.

This is intentionally simplified for educational and experimental use.
It is NOT a full reconstruction of the real connectome.
"""

import torch
import torch.nn as nn


class FlyCircuit(nn.Module):
    """
    A small recurrent circuit with fixed sparse connectivity.
    Only the final readout (and optional input gains) are trained.
    """

    def __init__(self, n_neurons: int = 64, n_inputs: int = 4, n_outputs: int = 2):
        super().__init__()
        self.n_neurons = n_neurons

        # Fixed recurrent weights (inspired by sparse biological connectivity)
        W = torch.randn(n_neurons, n_neurons) * 0.15
        # Make it sparse
        mask = torch.rand(n_neurons, n_neurons) > 0.85
        W = W * mask.float()
        # Remove self-connections
        W.fill_diagonal_(0.0)
        self.register_buffer("W", W)

        # Learnable input projection (sensory gains)
        self.input_proj = nn.Linear(n_inputs, n_neurons, bias=False)

        # Learnable readout (the main thing we train)
        self.readout = nn.Linear(n_neurons, n_outputs)

        # Simple leak parameter
        self.leak = 0.2

    def forward(self, x, h=None):
        """
        x: (batch, n_inputs)
        h: (batch, n_neurons) previous state
        """
        if h is None:
            h = torch.zeros(x.size(0), self.n_neurons, device=x.device)

        # Input drive
        drive = self.input_proj(x)

        # Recurrent dynamics (leaky)
        h = (1 - self.leak) * h + self.leak * torch.tanh(h @ self.W.T + drive)

        # Readout
        out = self.readout(h)
        return out, h

    def freeze_circuit(self):
        """Freeze the recurrent weights (keep only readout + input_proj trainable)."""
        self.W.requires_grad_(False)
