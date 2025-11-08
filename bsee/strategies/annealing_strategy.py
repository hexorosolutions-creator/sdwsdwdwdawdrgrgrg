"""
Simulated annealing strategy for BSEE.
"""

import random
import math
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class AnnealingStrategy(BaseStrategy):
    """Simulated annealing strategy."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize annealing strategy."""
        super().__init__(config)
        self.initial_temperature = config.get('initial_temperature', 100.0)
        self.cooling_rate = config.get('cooling_rate', 0.95)
        self.min_temperature = config.get('min_temperature', 0.1)
        self.current_temperature = self.initial_temperature

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation with temperature-dependent randomness."""
        operations = [
            ('xor_constant', {'constant': random.randint(1, 255)}),
            ('rotate_left', {'shift': random.randint(1, 7)}),
            ('move_to_front', {}),
            ('shuffle_bytes', {'seed': random.randint(0, 10000)})
        ]
        return random.choice(operations)

    def accept(self, new_state: State) -> bool:
        """Accept based on simulated annealing criteria."""
        if new_state.score > self.best_score:
            return True

        # Accept worse states with probability based on temperature
        if self.current_temperature > self.min_temperature:
            delta = new_state.score - self.best_score
            probability = math.exp(delta / self.current_temperature)
            if random.random() < probability:
                return True

        # Cool down
        self.current_temperature *= self.cooling_rate
        return False