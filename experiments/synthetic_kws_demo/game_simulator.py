"""
Lightweight 2D Grid Game Simulator for Voice-Command Prototype.
Simulates a grid world where a player character is steered by voice commands.
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

from experiments.synthetic_kws_demo.game_adapter import GameAction, AdaptedAction


@dataclass
class GameStepRecord:
    step_index: int
    command_label: str
    confidence: float
    action: GameAction
    delta: Tuple[int, int]
    player_pos: Tuple[int, int]
    event: str
    grid_render: str


class GridGameSimulator:
    """
    Self-contained 2D grid world:
      - Player: '@'
      - Goal: 'G'
      - Obstacles: '#'
      - Empty: '.'
    Bounds: [0..width-1, 0..height-1]
    """
    def __init__(
        self,
        width: int = 7,
        height: int = 7,
        initial_pos: Tuple[int, int] = (1, 5),
        goal_pos: Tuple[int, int] = (5, 1),
        obstacles: Optional[List[Tuple[int, int]]] = None
    ):
        self.width = width
        self.height = height
        self.initial_pos = initial_pos
        self.player_x, self.player_y = initial_pos
        self.goal_pos = goal_pos
        self.obstacles = set(obstacles) if obstacles is not None else {
            (2, 4), (2, 3), (4, 2), (4, 3)
        }
        self.history: List[GameStepRecord] = []
        self.goal_reached = False

    def reset(self):
        self.player_x, self.player_y = self.initial_pos
        self.history.clear()
        self.goal_reached = False

    def step(self, adapted_action: AdaptedAction) -> GameStepRecord:
        step_idx = len(self.history) + 1
        event = "STEADY"
        
        if adapted_action.is_active:
            new_x = self.player_x + adapted_action.delta_x
            new_y = self.player_y + adapted_action.delta_y
            
            # Boundary checks
            if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                event = f"BLOCKED_BY_BOUNDARY (Attempted out of bounds to {new_x}, {new_y})"
            elif (new_x, new_y) in self.obstacles:
                event = f"BLOCKED_BY_OBSTACLE (Hit obstacle at {new_x}, {new_y})"
            else:
                self.player_x = new_x
                self.player_y = new_y
                event = f"MOVED_{adapted_action.action.value} to ({new_x}, {new_y})"
                
                if (self.player_x, self.player_y) == self.goal_pos:
                    self.goal_reached = True
                    event += " [GOAL_REACHED!]"
        else:
            event = f"NO_ACTION ({adapted_action.notes})"
            
        render = self.render()
        record = GameStepRecord(
            step_index=step_idx,
            command_label=adapted_action.source_label,
            confidence=adapted_action.confidence,
            action=adapted_action.action,
            delta=(adapted_action.delta_x, adapted_action.delta_y),
            player_pos=(self.player_x, self.player_y),
            event=event,
            grid_render=render
        )
        self.history.append(record)
        return record

    def render(self) -> str:
        lines = []
        lines.append("+" + "---" * self.width + "+")
        for y in range(self.height):
            row_chars = []
            for x in range(self.width):
                if (x, y) == (self.player_x, self.player_y):
                    row_chars.append(" @ ")
                elif (x, y) == self.goal_pos:
                    row_chars.append(" G ")
                elif (x, y) in self.obstacles:
                    row_chars.append(" # ")
                else:
                    row_chars.append(" . ")
            lines.append("|" + "".join(row_chars) + "|")
        lines.append("+" + "---" * self.width + "+")
        return "\n".join(lines)
