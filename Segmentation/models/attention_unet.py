import torch
import torch.nn as nn
from typing import cast

from .unet import DoubleConv, Down, OutConv, Up


class AttentionGate(nn.Module):
    """Gate di attenzione additiva per filtrare le skip connections."""

    def __init__(self, skip_channels: int, gate_channels: int, inter_channels: int):
        super().__init__()
        self.W_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.W_x = nn.Conv2d(skip_channels, inter_channels, kernel_size=1)
        self.psi = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(inter_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self._init_open()

    def _init_open(self):
        with torch.no_grad():
            nn.init.normal_(self.W_g.weight, std=1e-3)
            nn.init.zeros_(self.W_g.bias)
            nn.init.normal_(self.W_x.weight, std=1e-3)
            nn.init.zeros_(self.W_x.bias)
            conv = cast(nn.Conv2d, self.psi[1])
            nn.init.zeros_(conv.weight)
            assert conv.bias is not None
            nn.init.constant_(conv.bias, 1.0)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.W_g(g) + self.W_x(x))
        with torch.no_grad():
            values = alpha.detach()
            self._alpha_stats = (
                float(values.mean().item()),
                float(values.min().item()),
                float(values.max().item()),
            )
        return x * alpha


class AttentionUp(Up):
    """Upscaling con attention gate sulla skip connection."""

    def __init__(self, in_channels: int, out_channels: int, groupnorm: bool = False):
        super().__init__(in_channels, out_channels, groupnorm=groupnorm)
        self.att_gate = AttentionGate(
            skip_channels=in_channels // 2,
            gate_channels=in_channels // 2,
            inter_channels=max(in_channels // 4, 1),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x2 = self.att_gate(x2, x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class AttentionUNet(nn.Module):
    """Modello Attention U-Net per segmentazione."""

    def __init__(self, n_channels: int, n_classes: int):
        super(AttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.in_conv = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = AttentionUp(1024, 512)
        self.up2 = AttentionUp(512, 256)
        self.up3 = AttentionUp(256, 128)
        self.up4 = AttentionUp(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def log_attention_stats(model: nn.Module, prefix: str = "") -> None:
    """Stampa statistiche delle mappe di attenzione per ogni gate."""
    idx = 0
    for module in model.modules():
        if isinstance(module, AttentionGate) and hasattr(module, "_alpha_stats"):
            mean_value, min_value, max_value = module._alpha_stats
            print(
                f"{prefix}AttnGate[{idx}]: mean={mean_value:.3f} "
                f"min={min_value:.3f} max={max_value:.3f}"
            )
            idx += 1