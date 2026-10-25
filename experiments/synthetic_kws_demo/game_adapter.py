"""
Game Control Adapter for Synthetic Vietnamese Keyword Spotting.
Decouples KWS classification results from game mechanics:
  - TRAI    -> MOVE_LEFT
  - PHAI    -> MOVE_RIGHT
  - LEN     -> JUMP (MOVE_UP)
  - XUONG   -> CROUCH (MOVE_DOWN)
  - UNKNOWN -> NO_ACTION
  - SILENCE -> NO_ACTION
Includes configurable experimental thresholding and debouncing.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

from experiments.synthetic_kws_demo.classifier import PredictionResult


class GameAction(str, Enum):
    MOVE_LEFT = "MOVE_LEFT"
    MOVE_RIGHT = "MOVE_RIGHT"
    JUMP = "JUMP"
    CROUCH = "CROUCH"
    NO_ACTION = "NO_ACTION"


ACTION_DISPLACEMENTS = {
    GameAction.MOVE_LEFT: (-1, 0),
    GameAction.MOVE_RIGHT: (1, 0),
    GameAction.JUMP: (0, -1),
    GameAction.CROUCH: (0, 1),
    GameAction.NO_ACTION: (0, 0)
}

COMMAND_TO_ACTION = {
    "TRAI": GameAction.MOVE_LEFT,
    "PHAI": GameAction.MOVE_RIGHT,
    "LEN": GameAction.JUMP,
    "XUONG": GameAction.CROUCH,
    "UNKNOWN": GameAction.NO_ACTION,
    "SILENCE": GameAction.NO_ACTION
}


@dataclass
class AdaptedAction:
    action: GameAction
    delta_x: int
    delta_y: int
    is_active: bool
    source_label: str
    confidence: float
    threshold: float
    debounced: bool
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "delta_x": self.delta_x,
            "delta_y": self.delta_y,
            "is_active": self.is_active,
            "source_label": self.source_label,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "debounced": self.debounced,
            "notes": self.notes
        }


class CommandToActionAdapter:
    """
    Translates raw KWS predictions into deterministic game actions.
    Enforces that UNKNOWN and SILENCE never trigger accidental character movement.
    """
    def __init__(
        self,
        confidence_threshold: float = 0.50,
        enable_debouncing: bool = False,
        debounce_cooldown_sec: float = 0.50
    ):
        self.confidence_threshold = confidence_threshold
        self.enable_debouncing = enable_debouncing
        self.debounce_cooldown_sec = debounce_cooldown_sec
        self._last_action: Optional[GameAction] = None
        self._last_timestamp: float = 0.0

    def adapt(self, prediction: PredictionResult, timestamp: float = 0.0) -> AdaptedAction:
        """
        Maps a PredictionResult into an AdaptedAction.
        """
        if not prediction.is_valid:
            return AdaptedAction(
                action=GameAction.NO_ACTION,
                delta_x=0,
                delta_y=0,
                is_active=False,
                source_label=prediction.predicted_label,
                confidence=0.0,
                threshold=self.confidence_threshold,
                debounced=False,
                notes=f"Invalid audio input: {prediction.error_message}"
            )
            
        raw_label = prediction.predicted_label
        confidence = prediction.confidence
        
        # 1. Non-target filter: UNKNOWN and SILENCE unconditionally yield NO_ACTION
        if raw_label in ["UNKNOWN", "SILENCE"]:
            return AdaptedAction(
                action=GameAction.NO_ACTION,
                delta_x=0,
                delta_y=0,
                is_active=False,
                source_label=raw_label,
                confidence=confidence,
                threshold=self.confidence_threshold,
                debounced=False,
                notes=f"Non-target class ({raw_label}) safely suppressed to NO_ACTION."
            )
            
        # 2. Experimental confidence threshold
        if confidence < self.confidence_threshold:
            return AdaptedAction(
                action=GameAction.NO_ACTION,
                delta_x=0,
                delta_y=0,
                is_active=False,
                source_label=raw_label,
                confidence=confidence,
                threshold=self.confidence_threshold,
                debounced=False,
                notes=(
                    f"Confidence {confidence:.3f} below experimental threshold {self.confidence_threshold:.3f}; "
                    "suppressed to NO_ACTION."
                )
            )
            
        target_action = COMMAND_TO_ACTION.get(raw_label, GameAction.NO_ACTION)
        
        # 3. Optional experimental debouncing rule
        if self.enable_debouncing and self._last_action == target_action:
            elapsed = timestamp - self._last_timestamp
            if elapsed < self.debounce_cooldown_sec:
                return AdaptedAction(
                    action=GameAction.NO_ACTION,
                    delta_x=0,
                    delta_y=0,
                    is_active=False,
                    source_label=raw_label,
                    confidence=confidence,
                    threshold=self.confidence_threshold,
                    debounced=True,
                    notes=(
                        f"Action {target_action.value} suppressed by experimental debouncing rule "
                        f"({elapsed:.3f}s < cooldown {self.debounce_cooldown_sec:.3f}s)."
                    )
                )
                
        # Commit active action
        self._last_action = target_action
        self._last_timestamp = timestamp
        dx, dy = ACTION_DISPLACEMENTS[target_action]
        
        return AdaptedAction(
            action=target_action,
            delta_x=dx,
            delta_y=dy,
            is_active=True,
            source_label=raw_label,
            confidence=confidence,
            threshold=self.confidence_threshold,
            debounced=False,
            notes=f"Command '{raw_label}' mapped to active action {target_action.value} ({dx:+d}, {dy:+d})."
        )

    def reset(self):
        """Resets adapter internal debouncing state."""
        self._last_action = None
        self._last_timestamp = 0.0


# Backward-compatible alias
GameControlAdapter = CommandToActionAdapter

