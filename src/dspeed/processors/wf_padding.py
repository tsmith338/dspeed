"""Processors for waveform padding."""

from __future__ import annotations

import numpy as np
from numba import guvectorize

from dspeed.errors import DSPFatal
from dspeed.utils import numba_defaults_kwargs as nb_kwargs


@guvectorize(
    [
        "void(float32[:], boolean, float32, float32[:])",
        "void(float64[:], boolean, float64, float64[:])",
    ],
    "(n),(),(),(m)",
    **nb_kwargs,
)
def wf_padding(
    w_in: np.ndarray,
    pad_baseline: bool,
    pad_value: float,
    # target_length: int,
    w_out: np.ndarray,
) -> None:
    """Pad waveform to a target length that is inferred from the difference between the input and output waveform.

    This processor pads the input waveform either at the baseline (beginning)
    or at the tail (end) to reach a specific target length.

    Parameters
    ----------
    w_in
        the input waveform.
    pad_baseline
        if True, pad at the beginning (baseline); if False, pad at the end (tail).
    pad_value
        value to use for padding (e.g., baseline mean for baseline padding,
        waveform max for tail padding).
    w_out
        padded waveform of length target_length.

    Note
    ----
    If target_length <= len(w_in), the waveform is truncated rather than padded.
    For baseline padding: [pad_values..., original_waveform]
    For tail padding: [original_waveform, pad_values...]

    JSON Configuration Example
    --------------------------

    .. code-block:: json

        "wf_padded": {
            "function": "wf_padding",
            "module": "dspeed.processors",
            "args": ["waveform", "True", "baseline", "2000", "wf_padded"],
            "unit": "ADC"
        }

    YAML Configuration Example
    --------------------------

    .. code-block:: yaml

        wf_padded:
          function: wf_padding
          module: dspeed.processors
          args:
            - waveform
            - True  # pad baseline
            - baseline  # padding value
            - 2000  # target length
            - wf_padded
          unit: ADC
    """

    pad_baseline = bool(pad_baseline)

    w_out[:] = np.nan

    target_length = len(w_out)

    if np.isnan(w_in).any():
        return

    if np.isnan(pad_value):
        raise DSPFatal("pad_value is NaN")

    if target_length <= 0:
        raise DSPFatal("target_length must be positive")

    if target_length != len(w_out):
        raise DSPFatal("target_length must equal output array length")

    input_length = len(w_in)

    if target_length <= input_length:
        if pad_baseline:
            w_out[:] = w_in[input_length - target_length :]
        else:
            w_out[:] = w_in[:target_length]
        return

    pad_samples = target_length - input_length

    if pad_baseline:
        w_out[:pad_samples] = pad_value
        w_out[pad_samples:] = w_in[:]
    else:
        w_out[:input_length] = w_in[:]
        w_out[input_length:] = pad_value
