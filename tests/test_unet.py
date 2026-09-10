"""Shape-check tests for src/models/unet.py.

These test the VANILLA U-Net's forward pass — the piece left as a TODO in
src/models/unet.py. They will fail with NotImplementedError until
VanillaUNet.forward is implemented; that's expected. Run them against your
own implementation as you build it out.

The pretrained-encoder tests below (test_pretrained_encoder_*) exercise the
segmentation_models_pytorch wrapping path, which is already fully
implemented — they should pass today, and serve as a shape-contract
reference for what the vanilla path should also satisfy.
"""

from __future__ import annotations

import pytest
import torch

from src.models.unet import PigUNet, VanillaUNet, build_model


@pytest.mark.parametrize(
    "batch_size,height,width",
    [
        (1, 64, 64),
        (2, 128, 128),
        (4, 160, 96),  # non-square, still divisible by 16
    ],
)
def test_vanilla_unet_forward_output_shape(batch_size, height, width):
    """Output spatial shape must match input; channel dim is out_channels."""
    model = VanillaUNet(in_channels=3, out_channels=1)
    model.eval()
    x = torch.randn(batch_size, 3, height, width)

    out = model(x)

    assert out.shape == (batch_size, 1, height, width)


def test_vanilla_unet_multi_channel_output():
    """out_channels other than 1 should be respected (e.g. for future multi-class use)."""
    model = VanillaUNet(in_channels=3, out_channels=3)
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    out = model(x)

    assert out.shape == (2, 3, 64, 64)


def test_vanilla_unet_via_pig_unet_wrapper():
    """PigUNet(encoder_name='vanilla') should delegate to VanillaUNet with matching output shape."""
    model = PigUNet(encoder_name="vanilla", in_channels=3, out_channels=1)
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    out = model(x)

    assert out.shape == (2, 1, 64, 64)


def test_vanilla_unet_output_is_finite():
    """Output should be finite raw logits (no NaN/Inf) — a basic sanity check
    independent of exact values, since VanillaUNet applies no final
    activation."""
    model = VanillaUNet(in_channels=3, out_channels=1)
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    out = model(x)

    assert torch.isfinite(out).all()


# --- Pretrained-encoder path: already implemented, should pass today ---


def test_pretrained_encoder_forward_shape():
    """encoder_weights=None avoids a network download while still exercising
    the segmentation_models_pytorch wrapping path end-to-end."""
    model = PigUNet(encoder_name="resnet34", encoder_weights=None, in_channels=3, out_channels=1)
    model.eval()
    x = torch.randn(2, 3, 128, 128)

    out = model(x)

    assert out.shape == (2, 1, 128, 128)


def test_build_model_from_config_dict():
    """build_model() should construct a model matching the config's model section."""
    config = {
        "model": {
            "encoder_name": "resnet34",
            "encoder_weights": None,
            "in_channels": 3,
            "out_channels": 1,
        }
    }
    model = build_model(config)
    assert isinstance(model, PigUNet)
    assert model.encoder_name == "resnet34"
