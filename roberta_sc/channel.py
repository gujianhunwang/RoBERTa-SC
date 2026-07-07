"""Channel models for RoBERTa-SC.

All channels operate on a complex-valued tensor of shape ``(B, L, D)`` where
``D`` is the number of complex symbols transmitted **per token**.  Every model
is differentiable so that gradients propagate end-to-end during training.

The signal-to-noise ratio (SNR) is defined with respect to the *average power
per complex symbol*.  The transmitter normalises the signal so that the mean
symbol power is unity; an SNR of ``snr_db`` therefore corresponds to a complex
Gaussian noise of variance ``N0 = 10**(-snr_db/10)`` (split evenly across the
real and imaginary parts).  This is the single, consistent SNR convention used
for every method reported in the paper.
"""

from __future__ import annotations

import torch


def _complex_awgn(signal: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Add circularly-symmetric complex Gaussian noise at the given SNR (dB)."""
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = 1.0 / snr_linear  # N0, since the signal power is normalised to 1
    noise_real = torch.randn_like(signal.real) * torch.sqrt(noise_power / 2.0)
    noise_imag = torch.randn_like(signal.imag) * torch.sqrt(noise_power / 2.0)
    return signal + torch.complex(noise_real, noise_imag)


def awgn_channel(signal: torch.Tensor, snr_db: torch.Tensor) -> torch.Tensor:
    """Additive White Gaussian Noise channel: ``o = s + n``."""
    return _complex_awgn(signal, snr_db)


def rayleigh_channel(signal: torch.Tensor, snr_db: torch.Tensor,
                     perfect_csi: bool = True, fading: str = "block") -> torch.Tensor:
    """Rayleigh fading channel: ``o = h * s + n``.

    The complex fading coefficient ``h ~ CN(0, 1)`` is drawn at a granularity
    set by ``fading``:

    * ``"block"`` -- one coefficient per transmission block (per sentence),
      i.e. *flat* (frequency-flat, slow) fading. This is the convention used
      for the Rayleigh curves reported in the paper.
    * ``"fast"`` -- an independent coefficient per token (fast fading); a
      strictly harder setting, also used during training for robustness.

    With ``perfect_csi=True`` the receiver applies zero-forcing equalisation
    (``o / h``), matching the perfect-CSI assumption stated in the paper.
    """
    B, L, _ = signal.shape
    Lh = L if fading == "fast" else 1
    sqrt2 = torch.sqrt(torch.tensor(2.0, device=signal.device))
    h_real = torch.randn(B, Lh, 1, device=signal.device) / sqrt2
    h_imag = torch.randn(B, Lh, 1, device=signal.device) / sqrt2
    h = torch.complex(h_real, h_imag)
    received = _complex_awgn(h * signal, snr_db)
    return received / h if perfect_csi else received


def apply_channel(signal: torch.Tensor, snr_db, channel: str = "awgn") -> torch.Tensor:
    """Dispatch to the requested channel model.

    ``snr_db`` may be a scalar or a ``(B, 1)`` tensor (one SNR per example,
    used during training to expose the model to a range of conditions).
    """
    if not torch.is_tensor(snr_db):
        snr_db = torch.tensor(float(snr_db), device=signal.device)
    if snr_db.dim() == 2:  # (B, 1) -> (B, 1, 1) for broadcasting over symbols
        snr_db = snr_db.unsqueeze(-1)
    channel = channel.lower()
    if channel == "awgn":
        return awgn_channel(signal, snr_db)
    if channel in ("rayleigh", "rayleigh_fading"):
        # Per-token (fast) fading with the average-SNR convention (E[|h|^2]=1,
        # noise referenced to unit transmit power); matches the training channel.
        return rayleigh_channel(signal, snr_db, fading="fast")
    raise ValueError(f"Unknown channel '{channel}' (expected 'awgn' or 'rayleigh').")
