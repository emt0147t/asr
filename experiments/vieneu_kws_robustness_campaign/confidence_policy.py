"""
Phase 31: Command-Confidence Rejection Policy Function.
Implements the validation-calibrated post-processing rejection rule:
  - If predicted class is one of the four target commands (TRAI, PHAI, LEN, XUONG)
    AND its softmax confidence is below threshold theta:
      override prediction to UNKNOWN (class 4).
  - Predictions already classified as UNKNOWN (4) or SILENCE (5) are untouched.
"""

import numpy as np
import torch

TARGET_COMMAND_INDICES = {0, 1, 2, 3}
UNKNOWN_INDEX = 4
SILENCE_INDEX = 5

def apply_confidence_policy(probabilities: np.ndarray, threshold: float) -> tuple:
    """
    Applies the validation-calibrated confidence policy to probability vectors.

    Args:
        probabilities: np.ndarray of shape (N, 6) or (6,) representing 6-class softmax probabilities.
        threshold: float in [0.0, 1.0], calibrated confidence cutoff for target commands.

    Returns:
        calibrated_predictions: np.ndarray of shape (N,) or int
        confidences: np.ndarray of shape (N,) or float
        rejections_applied: np.ndarray of shape (N,) of bools (True if overridden to UNKNOWN)
    """
    is_single = False
    if isinstance(probabilities, torch.Tensor):
        probabilities = probabilities.detach().cpu().numpy()

    if probabilities.ndim == 1:
        probabilities = probabilities[np.newaxis, :]
        is_single = True

    preds = np.argmax(probabilities, axis=1)
    confs = np.max(probabilities, axis=1)
    rejections = np.zeros(len(preds), dtype=bool)

    calibrated_preds = preds.copy()

    for i in range(len(preds)):
        # If predicted as a target command and confidence < threshold -> override to UNKNOWN
        if preds[i] in TARGET_COMMAND_INDICES and confs[i] < threshold:
            calibrated_preds[i] = UNKNOWN_INDEX
            rejections[i] = True

    if is_single:
        return int(calibrated_preds[0]), float(confs[0]), bool(rejections[0])
    return calibrated_preds, confs, rejections
