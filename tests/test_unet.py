import torch

from flood.unet import UNet, UNetConfig


def test_unet_returns_correct_shape():
    m = UNet(UNetConfig(in_channels=3, base_channels=8, n_levels=3))
    x = torch.randn(2, 3, 32, 32)
    out = m(x)
    assert out.shape == (2, 1, 32, 32)


def test_unet_handles_non_power_of_two_inputs():
    m = UNet(UNetConfig(in_channels=3, base_channels=8, n_levels=2))
    x = torch.randn(1, 3, 21, 21)
    out = m(x)
    assert out.shape[-2:] == (21, 21)


def test_unet_gradients_flow():
    m = UNet(UNetConfig(in_channels=2, base_channels=4, n_levels=2))
    x = torch.randn(1, 2, 16, 16)
    out = m(x)
    out.pow(2).mean().backward()
    g = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
            for p in m.parameters())
    assert g > 0
