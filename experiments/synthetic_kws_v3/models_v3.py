"""
Architectures for Synthetic Vietnamese KWS V3.
Includes:
  1. VietnameseTCResNet: Baseline TC-ResNet-14 with global temporal pooling (192 features).
  2. VietnameseTemporalTCResNet: TC-ResNet-14 with Temporal Split Pooling (early vs late intervals, 384 features).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.kws_architectures import TCResBlock


class VietnameseTCResNet(nn.Module):
    """
    Standard TC-ResNet-14 (identical to V2 baseline).
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(40, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer1 = TCResBlock(32, 32, kernel_size=9, stride=1)
        self.layer2 = TCResBlock(32, 48, kernel_size=9, stride=2)  # 101 -> 51
        self.layer3 = TCResBlock(48, 64, kernel_size=9, stride=2)  # 51 -> 26
        self.layer4 = TCResBlock(64, 96, kernel_size=9, stride=1)  # 26

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        mean_pool = torch.mean(out, dim=2)
        max_pool = torch.max(out, dim=2)[0]
        combined = torch.cat([mean_pool, max_pool], dim=1)

        combined = self.dropout(combined)
        return self.fc(combined)


class VietnameseTemporalTCResNet(nn.Module):
    """
    Temporal TC-ResNet-14 with Temporal Split Pooling.
    Preserves time-ordering information without increasing convolutional depth or receptive field.
    Splits the 26 final temporal frames into:
      - Global mean + max pooling (192)
      - Early half mean pooling (frames 0..12) (96)
      - Late half mean pooling (frames 13..25) (96)
    Total feature vector = 384 dimensions.
    Enables linear classifier to compare early-vs-late activation gradients (e.g. rising tone vs dipping tone).
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(40, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.layer1 = TCResBlock(32, 32, kernel_size=9, stride=1)
        self.layer2 = TCResBlock(32, 48, kernel_size=9, stride=2)  # 101 -> 51
        self.layer3 = TCResBlock(48, 64, kernel_size=9, stride=2)  # 51 -> 26
        self.layer4 = TCResBlock(64, 96, kernel_size=9, stride=1)  # 26

        self.dropout = nn.Dropout(p=dropout_p)
        # 96 (global mean) + 96 (global max) + 96 (early mean) + 96 (late mean) = 384
        self.fc = nn.Linear(384, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)  # (B, 96, 26)

        # Global pools
        mean_global = torch.mean(out, dim=2)
        max_global = torch.max(out, dim=2)[0]

        # Temporal split: early frames (0..12) and late frames (13..25)
        early_frames = out[:, :, :13]
        late_frames = out[:, :, 13:]

        mean_early = torch.mean(early_frames, dim=2)
        mean_late = torch.mean(late_frames, dim=2)

        combined = torch.cat([mean_global, max_global, mean_early, mean_late], dim=1)
        combined = self.dropout(combined)
        return self.fc(combined)
