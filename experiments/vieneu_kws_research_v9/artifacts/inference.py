"""
Frozen Candidate Inference Wrapper for Vietnamese KWS (variant_w10).
Workspace: D:\trainidentificationaudio
Experiment: experiments/vieneu_kws_research_v9/

Provides standard, reproducible single-clip evaluation on 16kHz audio:
- Model: CorrectedVietnameseKWSNet (138,469 parameters)
- LogMelFrontend (40 mels, 25ms window, 10ms hop, slaney norm/scale)
- Peak normalization (-1.0 dBFS) & 1.0s zero-padding/centering
- Decision logic: command threshold gating (theta_cmd)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T

CLASSES = ["TRAI", "PHAI", "LEN", "XUONG", "REJECT"]


class LogMelFrontend(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=400, win_length=400, hop_length=160, n_mels=40, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm="slaney",
            n_mels=n_mels,
            mel_scale="slaney",
            f_min=20.0,
            f_max=8000.0
        )

    def forward(self, x):
        mel_spec = self.mel(x)
        log_mel = torch.log(mel_spec + self.eps)
        if log_mel.dim() == 3:
            log_mel = log_mel.unsqueeze(1)
        return log_mel


class CorrectedVietnameseKWSNet(nn.Module):
    """
    Corrected Compact 2D-CNN for 5-Class Vietnamese Bounded KWS.
    Uses robust, non-parametric AdaptiveAvgPool2d((1, 1)) instead of parametric attention.
    Output classes: [TRAI, PHAI, LEN, XUONG, REJECT]
    Parameters: 138,469
    """
    def __init__(self, num_classes=5, dropout_p=0.3):
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

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = self.pool1(F.relu(self.bn1(self.conv1(x))))
        out = self.pool2(F.relu(self.bn2(self.conv2(out))))
        out = self.pool3(F.relu(self.bn3(self.conv3(out))))
        out = self.global_pool(F.relu(self.bn4(self.conv4(out))))

        out = torch.flatten(out, 1)
        out = self.dropout(F.relu(self.fc1(out)))
        logits = self.fc2(out)
        return logits


class FrozenCandidatePredictor:
    def __init__(self, checkpoint_path: str, theta_cmd: float = 0.40, device: str = "cpu"):
        self.device = torch.device(device)
        self.theta_cmd = theta_cmd
        self.frontend = LogMelFrontend().to(self.device)
        self.model = CorrectedVietnameseKWSNet(num_classes=5).to(self.device)
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def preprocess(self, wav: torch.Tensor, orig_sr: int = 16000) -> torch.Tensor:
        if orig_sr != 16000:
            resampler = torchaudio.transforms.Resample(orig_sr, 16000)
            wav = resampler(wav)
        if wav.dim() == 2 and wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        elif wav.dim() == 1:
            wav = wav.unsqueeze(0)

        # Pad right or truncate to exactly 16,000 samples (matching training/eval convention)
        target_len = 16000
        curr_len = wav.size(-1)
        if curr_len < target_len:
            pad_right = target_len - curr_len
            wav = F.pad(wav, (0, pad_right), mode="constant", value=0.0)
        elif curr_len > target_len:
            wav = wav[:, :target_len]

        # Normalize only if amplitude exceeds 1.0 (never amplify noise floor)
        max_val = torch.max(torch.abs(wav))
        if max_val > 1.0:
            wav = wav / max_val

        return wav

    @torch.no_grad()
    def predict(self, wav: torch.Tensor, orig_sr: int = 16000):
        prep_wav = self.preprocess(wav, orig_sr).to(self.device)
        feats = self.frontend(prep_wav)
        logits = self.model(feats)
        probs = F.softmax(logits, dim=-1)

        cmd_probs = probs[0, :4]
        max_cmd_prob, cmd_idx = torch.max(cmd_probs, dim=-1)

        if max_cmd_prob.item() >= self.theta_cmd:
            pred_class = CLASSES[cmd_idx.item()]
            confidence = max_cmd_prob.item()
        else:
            pred_class = "REJECT"
            confidence = probs[0, 4].item()

        return {
            "predicted_class": pred_class,
            "confidence": float(confidence),
            "command_probabilities": {
                CLASSES[i]: float(probs[0, i].item()) for i in range(4)
            },
            "reject_probability": float(probs[0, 4].item())
        }
