"""
Vietnamese Keyword Spotting Model Architectures.
Provides:
  1. VietnameseKWSNet: Baseline 2D CNN with global average pooling (138k params).
  2. VietnameseTemporalKWSNet: 2D-CNN with temporal preservation & attentive pooling (~185k params).
  3. VietnameseTCResNet: Temporal 1D Convolutional ResNet (TC-ResNet-14, ~110k params).
All models take input (Batch, 1, 40, 101) or (Batch, 40, 101) and output logits (Batch, 6).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================
# 1. Baseline 2D-CNN
# =====================================================================
class VietnameseKWSNet(nn.Module):
    """
    Baseline 2D-CNN.
    Averages over entire time & frequency axis via AdaptiveAvgPool2d((1, 1)).
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.global_pool(F.relu(self.bn4(self.conv4(out))))
        out = torch.flatten(out, 1)
        out = self.dropout(F.relu(self.fc1(out)))
        return self.fc2(out)


# =====================================================================
# 2. Temporal-Preserving 2D-CNN with Attentive Temporal Pooling
# =====================================================================
class TemporalAttention(nn.Module):
    """
    Learns temporal importance weights over frames.
    Particularly effective for capturing transient consonant bursts (e.g. affricate /tɕ/).
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, Time, Features)
        scores = self.attn_net(x)  # (Batch, Time, 1)
        weights = F.softmax(scores, dim=1)  # (Batch, Time, 1)
        context = torch.sum(weights * x, dim=1)  # (Batch, Features)
        return context


class VietnameseTemporalKWSNet(nn.Module):
    """
    Temporal-Preserving 2D-CNN.
    Key differences from baseline:
      1. Preserves temporal frame resolution by pooling frequency selectively:
         Stage 1: pool (2, 2) -> (20, 50)
         Stage 2: pool (2, 1) -> (10, 50)  <-- Keeps all 50 time frames
         Stage 3: pool (2, 2) -> (5, 25)
      2. Frequency-only pooling AdaptiveAvgPool2d((1, None)) -> (128, 1, 25)
      3. 1D Temporal convolution across time frames to capture pitch contours (tones)
      4. Attentive Temporal Pooling + Max Pooling (captures transient affricates)
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        # 2D Spectrogram Feature Extraction
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # (40, 101) -> (20, 50)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # (20, 50) -> (10, 50)

        self.conv3 = nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # (10, 50) -> (5, 25)

        self.conv4 = nn.Conv2d(96, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))  # (5, 25) -> (1, 25)

        # 1D Temporal Processing along the 25 time frames
        self.temporal_conv = nn.Conv1d(128, 128, kernel_size=5, padding=2, bias=False)
        self.bn_temp = nn.BatchNorm1d(128)

        # Attentive temporal pooling
        self.temporal_attn = TemporalAttention(in_features=128)

        # Classifier: combines attention context (128) + temporal max-pool (128) = 256
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.freq_pool(F.relu(self.bn4(self.conv4(out))))  # (B, 128, 1, 25)
        
        # Squeeze frequency -> (B, 128, 25)
        out = out.squeeze(2)
        
        # Temporal convolution with residual connection
        res = out
        out = F.relu(self.bn_temp(self.temporal_conv(out))) + res
        
        # Temporal representations: (B, 25, 128)
        seq = out.permute(0, 2, 1)
        
        # 1. Attentive context vector
        context_att = self.temporal_attn(seq)  # (B, 128)
        # 2. Peak activation across time (detects burst transients)
        context_max = torch.max(seq, dim=1)[0]  # (B, 128)
        
        # Combined feature vector (256)
        combined = torch.cat([context_att, context_max], dim=1)
        
        combined = self.dropout(F.relu(self.fc1(combined)))
        logits = self.fc2(combined)
        return logits


# =====================================================================
# 3. Temporal 1D Convolutional ResNet (TC-ResNet-14)
# =====================================================================
class TCResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 9, stride: int = 1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out + res)
        return out


class VietnameseTCResNet(nn.Module):
    """
    TC-ResNet-14 adapted for Vietnamese Keyword Spotting.
    Processes the 40 mel bands directly across 101 time steps with 1D residual blocks.
    Known for superior pitch trajectory modeling and micro-KWS efficiency.
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        # Input: (B, 40, 101)
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
        # Combined mean + max pooling across time = 96 * 2 = 192
        self.fc = nn.Linear(192, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)  # (B, 1, 40, 101) -> (B, 40, 101)
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Global average + max pool across time
        mean_pool = torch.mean(out, dim=2)
        max_pool = torch.max(out, dim=2)[0]
        combined = torch.cat([mean_pool, max_pool], dim=1)

        combined = self.dropout(combined)
        logits = self.fc(combined)
        return logits


def get_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    dummy_x = torch.randn(4, 1, 40, 101)
    
    m1 = VietnameseKWSNet()
    m2 = VietnameseTemporalKWSNet()
    m3 = VietnameseTCResNet()

    print(f"VietnameseKWSNet (Baseline):          {get_param_count(m1):,} params, out: {m1(dummy_x).shape}")
    print(f"VietnameseTemporalKWSNet (Improved):  {get_param_count(m2):,} params, out: {m2(dummy_x).shape}")
    print(f"VietnameseTCResNet (TC-ResNet-14):   {get_param_count(m3):,} params, out: {m3(dummy_x).shape}")
