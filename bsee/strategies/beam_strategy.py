"""
Beam search strategy for BSEE.
"""

import random
import copy
import yaml
import math
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class BeamState:
    """Represents a state in the beam search."""

    def __init__(self, state: State, operations_applied: List[Tuple[str, Dict[str, Any]]], score: float):
        """Initialize a beam state."""
        self.state = state
        self.operations_applied = operations_applied.copy()
        self.score = score
        self.age = 0
        self.diversity_score = 0.0

    def __lt__(self, other):
        """Less-than comparison for sorting."""
        return self.score > other.score  # Higher scores are better

    def copy(self) -> 'BeamState':
        """Create a copy of this beam state."""
        return BeamState(
            copy.deepcopy(self.state),
            self.operations_applied.copy(),
            self.score
        )

    def calculate_diversity(self, other_states: List['BeamState']) -> float:
        """Calculate diversity score compared to other states."""
        if not other_states:
            return 1.0

        diversity_scores = []
        for other in other_states:
            if other is not self:
                # Simple diversity based on operation sequence difference
                ops_diff = len(set(self.operations_applied) - set(other.operations_applied))
                max_ops = max(len(self.operations_applied), len(other.operations_applied))
                diversity = ops_diff / max(1, max_ops)
                diversity_scores.append(diversity)

        return sum(diversity_scores) / len(diversity_scores) if diversity_scores else 0.0


class BeamStrategy(BaseStrategy):
    """Beam search strategy that maintains multiple candidate states."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize beam strategy."""
        super().__init__(config)
        self.beam_width = config.get('beam_width', 5)
        self.beam: List[Tuple[State, float]] = []  # (state, score)

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation for beam search."""
        operations = [
            ('xor_constant', {'constant': random.randint(1, 255)}),
            ('rotate_left', {'shift': random.randint(1, 7)}),
            ('move_to_front', {}),
            ('shuffle_bytes', {'seed': random.randint(0, 10000)})
        ]
        return random.choice(operations)

    def accept(self, new_state: State) -> bool:
        """Accept state based on beam criteria."""
        return new_state.score > self.best_score