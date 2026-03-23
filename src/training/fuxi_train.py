#!/usr/bin/env python3
"""
Legacy compatibility layer for FuXi pretraining.

This module keeps old imports working while delegating all training logic to
`pretraining.py`, which is the maintained implementation.

Kept public symbols:
- `FuXiZarrDataset`
- `LatitudeWeightedL1Loss`
- `main()`
"""

from __future__ import annotations

import os
import sys
from typing import Optional, Sequence

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

try:
    from .loss import LatitudeWeightedL1Loss
    from .pretraining import (
        DEFAULT_PRESSURE_LEVELS,
        DEFAULT_PRESSURE_VARS,
        DEFAULT_SURFACE_VARS,
        FuXiZarrDataset as _FuXiZarrDatasetV2,
        main as _pretraining_main,
        resolve_variable_names,
    )
except ImportError:
    from loss import LatitudeWeightedL1Loss
    from pretraining import (
        DEFAULT_PRESSURE_LEVELS,
        DEFAULT_PRESSURE_VARS,
        DEFAULT_SURFACE_VARS,
        FuXiZarrDataset as _FuXiZarrDatasetV2,
        main as _pretraining_main,
        resolve_variable_names,
    )


class FuXiZarrDataset(_FuXiZarrDatasetV2):
    """
    Backward-compatible dataset wrapper.

    Old code usually called:
        FuXiZarrDataset(zarr_path, history_steps=2, time_start=..., time_end=...)

    The maintained dataset in `pretraining.py` also needs explicit variable and
    pressure-level lists. This wrapper provides paper/protocol defaults and
    forwards to the maintained implementation.
    """

    def __init__(
        self,
        zarr_path: str,
        history_steps: int = 2,
        time_start: Optional[str] = None,
        time_end: Optional[str] = None,
        mean: Optional[torch.Tensor] = None,
        std: Optional[torch.Tensor] = None,
        stats_subsample: int = 200,
        pressure_vars: Optional[Sequence[str]] = None,
        surface_vars: Optional[Sequence[str]] = None,
        pressure_levels: Optional[Sequence[int]] = None,
    ) -> None:
        pressure_vars = list(pressure_vars) if pressure_vars is not None else list(DEFAULT_PRESSURE_VARS)
        surface_vars = list(surface_vars) if surface_vars is not None else list(DEFAULT_SURFACE_VARS)
        pressure_levels = (
            [int(v) for v in pressure_levels]
            if pressure_levels is not None
            else list(DEFAULT_PRESSURE_LEVELS)
        )

        resolved_pressure, resolved_surface, _ = resolve_variable_names(
            zarr_path=zarr_path,
            requested_pressure_vars=pressure_vars,
            requested_surface_vars=surface_vars,
        )

        super().__init__(
            zarr_path=zarr_path,
            pressure_vars=resolved_pressure,
            surface_vars=resolved_surface,
            pressure_levels=pressure_levels,
            history_steps=history_steps,
            time_start=time_start or "1979-01-01",
            time_end=time_end or "2022-12-31",
            mean=mean,
            std=std,
            stats_samples=stats_subsample,
        )


def main() -> None:
    """Entrypoint preserved for `python fuxi_train.py`."""
    _pretraining_main()


if __name__ == "__main__":
    main()
