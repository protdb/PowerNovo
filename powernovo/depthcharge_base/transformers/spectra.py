"""Tranformer models to handle mass spectra."""
from collections.abc import Callable
from typing import Union

import torch

from powernovo.depthcharge_base.encoders.sinusoidal import PeakEncoder


class SpectrumTransformerEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 1,
        dropout: float = 0,
        peak_encoder: Union[PeakEncoder,  Callable, bool] = True,
    ) -> None:
        super().__init__()

        self.latent_spectrum = torch.nn.Parameter(torch.randn(1, 1, d_model))

        if callable(peak_encoder):
            self.peak_encoder = peak_encoder
        elif peak_encoder:
            self.peak_encoder = PeakEncoder(d_model)
        else:
            self.peak_encoder = torch.nn.Linear(2, d_model)

        # The Transformer layers:
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout,
        )

        self.transformer_encoder = torch.nn.TransformerEncoder(
            layer,
            num_layers=n_layers,
        )

    def forward(
        self,
        spectra: torch.Tensor,
    ) -> tuple:
        zeros = ~spectra.sum(dim=2).bool()
        mask = [
            torch.tensor([[False]] * spectra.shape[0]).type_as(zeros),
            zeros,
        ]
        mask = torch.cat(mask, dim=1)
        peaks = self.peak_encoder(spectra)

        # Add the spectrum representation to each input:
        latent_spectra = self.latent_spectrum.expand(peaks.shape[0], -1, -1)

        peaks = torch.cat([latent_spectra, peaks], dim=1)
        return self.transformer_encoder(peaks, src_key_padding_mask=mask), mask

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
