"""Encoder-swappable U-Net for pig/dorsal-region segmentation.

Two code paths, selected by config.model.encoder_name:
- "vanilla" (default): a from-scratch U-Net encoder/decoder, defined here
  as VanillaUNet. forward() — the skip-connection wiring — is left for the
  user to implement; see the TODO in VanillaUNet.forward below. This is
  the intended learning exercise.
- anything else (e.g. "timm-mobilenetv3_small_100", "resnet34"): delegates
  to segmentation_models_pytorch's Unet with that pretrained encoder. This
  path is fully implemented — swapping encoders is a config-only change,
  no code changes.

Expected shapes throughout: input (B, in_channels, H, W) float tensor,
output (B, out_channels, H, W) raw logits (no sigmoid applied — losses.py
and metrics.py expect logits and apply sigmoid/threshold themselves).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover - exercised only when smp isn't installed
    smp = None


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x2. Standard U-Net building block.

    Input: (B, in_channels, H, W). Output: (B, out_channels, H, W) — spatial
    dims unchanged (padding=1 convs preserve H, W).
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VanillaUNet(nn.Module):
    """From-scratch U-Net encoder/decoder (Ronneberger et al. 2015 pattern).

    Architecture (standard 4-level U-Net):
      Encoder: DoubleConv(in_channels->64) -> pool -> DoubleConv(64->128) -> pool
               -> DoubleConv(128->256) -> pool -> DoubleConv(256->512) -> pool
      Bottleneck: DoubleConv(512->1024)
      Decoder: 4x [upsample -> concat skip -> DoubleConv], mirroring the
               encoder channel counts back down to 64
      Head: Conv2d(64 -> out_channels, kernel_size=1)

    The constructor builds every submodule listed above (encoder blocks,
    pooling, bottleneck, decoder up-convs + blocks, output head) — all
    present and ready to use. forward(), i.e. the skip-connection wiring
    between encoder and decoder, is left as a TODO for the user to
    implement.

    Expected shapes: input (B, in_channels, H, W); output (B, out_channels,
    H, W). H and W should be divisible by 16 (4 pooling stages) to avoid
    shape mismatches when concatenating skip connections; the training
    pipeline resizes inputs to a fixed size (configs/segmentation.yaml) so
    this holds by construction.
    """

    ENCODER_CHANNELS = (64, 128, 256, 512)
    BOTTLENECK_CHANNELS = 1024

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder: one DoubleConv per level; downsampled by max-pool between levels.
        self.encoder_blocks = nn.ModuleList()
        enc_in = in_channels
        for enc_out in self.ENCODER_CHANNELS:
            self.encoder_blocks.append(DoubleConv(enc_in, enc_out))
            enc_in = enc_out
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck.
        self.bottleneck = DoubleConv(self.ENCODER_CHANNELS[-1], self.BOTTLENECK_CHANNELS)

        # Decoder: one transposed-conv upsample + DoubleConv per level, mirroring the encoder.
        self.up_convs = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        dec_in = self.BOTTLENECK_CHANNELS
        for dec_out in reversed(self.ENCODER_CHANNELS):
            self.up_convs.append(nn.ConvTranspose2d(dec_in, dec_out, kernel_size=2, stride=2))
            # Decoder DoubleConv input channels = upsampled (dec_out) + skip (dec_out) = 2 * dec_out
            self.decoder_blocks.append(DoubleConv(dec_out * 2, dec_out))
            dec_in = dec_out

        # Output head: 1x1 conv to out_channels, no activation (raw logits).
        self.out_conv = nn.Conv2d(self.ENCODER_CHANNELS[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the U-Net forward pass.

        Expected: x of shape (B, in_channels, H, W) -> output of shape
        (B, out_channels, H, W), raw logits.

        Implementation notes for the skip-connection wiring:
          - Save each encoder block's output BEFORE pooling — these are the
            skip connections, needed in the decoder at matching resolution.
            Apply self.pool after saving each skip, then feed the pooled
            tensor into the next encoder block.
          - After the last encoder block + pool, run self.bottleneck.
          - For each decoder level (self.up_convs[i], self.decoder_blocks[i]):
            upsample the current tensor with up_convs[i], concatenate with
            the matching skip connection along the channel dimension
            (torch.cat([upsampled, skip], dim=1)), then run decoder_blocks[i].
          - Skip connections are consumed in reverse order of creation: the
            last encoder skip (highest-level features) pairs with the FIRST
            decoder level.
          - Finish with self.out_conv on the final decoder output.
        """
        # TODO(timotius): implement the U-Net forward pass (see notes above)
        raise NotImplementedError("TODO(timotius): implement the U-Net forward pass")


class PigUNet(nn.Module):
    """Encoder-swappable segmentation model — what training/inference code imports.

    encoder_name == "vanilla" -> wraps VanillaUNet (forward is the TODO
    stub above). Any other encoder_name -> wraps
    segmentation_models_pytorch.Unet with that pretrained encoder (fully
    implemented today; e.g. "timm-mobilenetv3_small_100", "resnet34").

    Swapping encoders is a config-only change (configs/segmentation.yaml
    model.encoder_name) — no code changes required either way.
    """

    def __init__(
        self,
        encoder_name: str = "vanilla",
        encoder_weights: Optional[str] = None,
        in_channels: int = 3,
        out_channels: int = 1,
    ):
        super().__init__()
        self.encoder_name = encoder_name

        if encoder_name == "vanilla":
            self.model: nn.Module = VanillaUNet(in_channels=in_channels, out_channels=out_channels)
        else:
            if smp is None:
                raise ImportError(
                    "segmentation_models_pytorch is required for non-vanilla encoders "
                    "(pip install segmentation-models-pytorch)"
                )
            self.model = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                in_channels=in_channels,
                classes=out_channels,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_channels, H, W) -> (B, out_channels, H, W) raw logits."""
        return self.model(x)


def build_model(config: dict) -> PigUNet:
    """Construct a PigUNet from a config dict (the `model` section of configs/segmentation.yaml)."""
    model_cfg = config["model"]
    return PigUNet(
        encoder_name=model_cfg.get("encoder_name", "vanilla"),
        encoder_weights=model_cfg.get("encoder_weights"),
        in_channels=model_cfg.get("in_channels", 3),
        out_channels=model_cfg.get("out_channels", 1),
    )
