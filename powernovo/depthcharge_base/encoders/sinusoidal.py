
import math

import einops
import numpy as np
import torch


class FloatEncoder(torch.nn.Module):

    def __init__(
        self,
        d_model: int,
        min_wavelength: float = 0.001,
        max_wavelength: float = 10000,
        learnable_wavelengths: bool = False,
    ) -> None:
        super().__init__()

        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        self.learnable_wavelengths = learnable_wavelengths

        # Get dimensions for equations:
        d_sin = math.ceil(d_model / 2)
        d_cos = d_model - d_sin

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, d_model).float() - d_sin) / (d_cos - 1)
        sin_term = base * (scale**sin_exp)
        cos_term = base * (scale**cos_exp)

        if not self.learnable_wavelengths:
            self.register_buffer("sin_term", sin_term)
            self.register_buffer("cos_term", cos_term)
        else:
            self.sin_term = torch.nn.Parameter(sin_term)
            self.cos_term = torch.nn.Parameter(cos_term)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        sin_mz = torch.sin(X[:, :, None] / self.sin_term)
        cos_mz = torch.cos(X[:, :, None] / self.cos_term)
        return torch.cat([sin_mz, cos_mz], axis=-1)


class PeakEncoder(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        min_mz_wavelength: float = 0.001,
        max_mz_wavelength: float = 10000,
        min_intensity_wavelength: float = 1e-6,
        max_intensity_wavelength: float = 1,
        learnable_wavelengths: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.learnable_wavelengths = learnable_wavelengths

        self.mz_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )

        self.int_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_intensity_wavelength,
            max_wavelength=max_intensity_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )
        self.combiner = torch.nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        encoded = torch.cat(
            [
                self.mz_encoder(X[:, :, 0]),
                self.int_encoder(X[:, :, 1]),
            ],
            dim=2,
        )

        return self.combiner(encoded)


class PositionalEncoder(FloatEncoder):
    def __init__(
        self,
        d_model: int,
        min_wavelength: float = 1.0,
        max_wavelength: float = 1e5,
    ) -> None:
        super().__init__(
            d_model=d_model,
            min_wavelength=min_wavelength,
            max_wavelength=max_wavelength,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(X.shape[1]).type_as(self.sin_term)
        pos = einops.repeat(pos, "n -> b n", b=X.shape[0])
        sin_in = einops.repeat(pos, "b n -> b n f", f=len(self.sin_term))
        cos_in = einops.repeat(pos, "b n -> b n f", f=len(self.cos_term))

        sin_pos = torch.sin(sin_in / self.sin_term)
        cos_pos = torch.cos(cos_in / self.cos_term)
        encoded = torch.cat([sin_pos, cos_pos], axis=2)
        return encoded + X
