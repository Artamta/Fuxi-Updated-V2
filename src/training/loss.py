"""
Latitude-weighted L1 loss – paper Eq. (2).

    L = (1 / C) Σ_c Σ_i Σ_j  a_i · |pred_{c,i,j} − truth_{c,i,j}|  /  (H · W)

where a_i = cos(latitude_i), giving higher weight near the equator and
lower weight near the poles (accounts for grid-cell area on a
latitude–longitude grid).
"""

import numpy as np
import torch
import torch.nn as nn


class LatitudeWeightedL1Loss(nn.Module):
    """
    Latitude-weighted mean absolute error.

    Parameters
    ----------
    num_lat : int
        Number of latitude grid points.
    lat_range : tuple
        (south, north) in degrees.  Default: (−90, 90).
    """

    def __init__(self, num_lat: int = 121, lat_range: tuple = (-90.0, 90.0)):
        super().__init__()
        lats = np.linspace(lat_range[0], lat_range[1], num_lat)
        weights = np.cos(np.deg2rad(lats))
        weights = weights / weights.mean()  # normalize so mean weight = 1
        # Shape: (1, 1, H, 1) for broadcasting with (B, C, H, W)
        w = torch.from_numpy(weights).float().reshape(1, 1, -1, 1)
        self.register_buffer("weights", w)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred, target : (B, C, H, W)
        Returns:
            scalar loss
        """
        return (torch.abs(pred - target) * self.weights).mean()
