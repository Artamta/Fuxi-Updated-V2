import torch

from src.models.fuxi_model import make_fuxi
from src.training.loss import LatitudeWeightedL1Loss


def test_mini_model_forward():
    model = make_fuxi(
        preset="mini",
        num_variables=70,
        input_height=48,
        input_width=96,
        patch_size=(2, 4, 4),
    )
    x = torch.randn(1, 70, 2, 48, 96)
    with torch.no_grad():
        y = model(x)
    assert y.shape == (1, 70, 48, 96)


def test_mini_model_loss():
    model = make_fuxi(
        preset="mini",
        num_variables=70,
        input_height=48,
        input_width=96,
        patch_size=(2, 4, 4),
    )
    x = torch.randn(1, 70, 2, 48, 96)
    target = torch.zeros(1, 70, 48, 96)
    criterion = LatitudeWeightedL1Loss(num_lat=48)
    with torch.no_grad():
        pred = model(x)
        loss = criterion(pred, target)
    assert torch.isfinite(loss)
