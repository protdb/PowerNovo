from collections.abc import Callable
from functools import wraps

import numpy as np

from powernovo.depthcharge_base.utils.primitives import MassSpectrum


def scale_to_unit_norm(spectrum: MassSpectrum) -> MassSpectrum:
    """Scale intensities to unit norm."""

    spectrum._inner._intensity = (
        spectrum.intensity / np.sqrt(spectrum.intensity**2).sum()
    )
    return spectrum


def _spectrum_utils_fn(func: str) -> Callable:
    """Wrap spectrum_utils.spectrum.MmsmsSpectrum preprocessing methods."""

    @wraps(getattr(MassSpectrum, func))
    def wrapper(
        *args: tuple,
        **kwargs: dict,
    ) -> Callable:
        """Wrapper for spectrum_utils MsmsSpectrum methods.

        Parameters
        ----------
        func: Callable
            The preprocessing function. This should exactly match a
            MsmsSpectrum method.
        *args: tuple
            Arguments that are passed to the MsmsSpectrum method.
        **kwargs : dict
            Keyword arguments that are passed to the MsmsSpectrum method.

        Returns
        -------
        Callable
            A valid deptcharge preprocessing function.
        """

        @wraps(wrapper)
        def preprocess(spec: MassSpectrum) -> MassSpectrum:
            """The wrapped preprocessing function.

            Parameters
            ----------
            spec : MassSpectrum
                The mass spectrum to preprocess

            Returns
            -------
            MassSpectrum
                The processed mass spectrum.
            """
            # Call the spectrum_utils method:
            getattr(spec, func)(*args, **kwargs)
            return spec

        return preprocess

    return wrapper


def _add_spectrum_utils_methods() -> None:
    """Update this module with the spectrum_utils MsmsSpectrum methods."""
    sus_methods = [
        "filter_intensity",
        "remove_precursor_peak",
        "round",
        "scale_intensity",
        "set_mz_range",
    ]

    for method in sus_methods:
        globals()[method] = _spectrum_utils_fn(method)


_add_spectrum_utils_methods()
