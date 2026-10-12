"""
Lightweight Log-Mel 2D-CNN Architecture for On-Device Vietnamese KWS.
Optimized for mobile inference on Android and iOS (low parameter count, low latency).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VietnameseKWSNet(nn.Module):
    """
    Compact 2D-CNN for 6-class Vietnamese keyword spotting.
    Input: Log-Mel Spectrogram (Batch, 1, 40, 101)
    Output: Logits (Batch, 6)
      0 = TRAI
      1 = PHAI
      2 = LEN
      3 = XUONG
      4 = UNKNOWN
      5 = SILENCE
    """
    def __init__(self, num_classes: int = 6, dropout_p: float = 0.3):
        super().__init__()
        # Conv Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # (40, 101) -> (20, 50)

        # Conv Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # (20, 50) -> (10, 25)

        # Conv Block 3
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # (10, 25) -> (5, 12)

        # Conv Block 4
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))      # (5, 12) -> (1, 1)

        # Classification Head
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape check
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 40, 101) -> (B, 1, 40, 101)

        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.global_pool(F.relu(self.bn4(self.conv4(out))))
        
        out = torch.flatten(out, 1)
        out = self.dropout(F.relu(self.fc1(out)))
        logits = self.fc2(out)
        return logits


def get_model_param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = VietnameseKWSNet()
    dummy_input = torch.randn(2, 1, 40, 101)
    out = model(dummy_input)
    print("Model Output Shape:", out.shape)
    print("Total Trainable Parameters:", get_model_param_count(model))
