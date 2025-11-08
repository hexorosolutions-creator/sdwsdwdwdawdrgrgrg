"""
Monte Carlo Tree Search strategy for BSEE.
"""

import random
import math
import copy
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class TreeNode:
    """Node in the Monte Carlo Tree Search."""

    def __init__(self, state: State, parent: Optional['TreeNode'] = None, operation: Optional[Tuple[str, Dict[str, Any]]] = None):
        """Initialize a tree node."""
        self.state = state
        self.parent = parent
        self.operation = operation  # (operation_name, operation_params) that led to this node
        self.children = {}  # operation_name -> TreeNode
        self.visits = 0
        self.value = 0.0
        self.untried_operations = []  # List of operations not yet tried

    def is_fully_expanded(self) -> bool:
        """Check if all operations have been tried from this node."""
        return len(self.untried_operations) == 0

    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.state.is_terminal()

    def get_best_child(self, exploration_constant: float = 1.4) -> 'TreeNode':
        """Get the best child using UCT formula."""
        best_child = None
        best_score = float('-inf')

        for child in self.children.values():
            if child.visits == 0:
                return child  # Unvisited child gets priority

            # UCT formula: value/visits + exploration_constant * sqrt(ln(parent.visits)/visits)
            exploitation = child.value / child.visits
            exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
            uct_score = exploitation + exploration

            if uct_score > best_score:
                best_score = uct_score
                best_child = child

        return best_child

    def update(self, result: float) -> None:
        """Update node statistics with simulation result."""
        self.visits += 1
        self.value += result

    def get_most_visited_child(self) -> 'TreeNode':
        """Get the child with the most visits."""
        if not self.children:
            return self
        return max(self.children.values(), key=lambda child: child.visits)

    def get_highest_value_child(self) -> 'TreeNode':
        """Get the child with the highest average value."""
        if not self.children:
            return self
        return max(self.children.values(), key=lambda child: child.value / child.visits if child.visits > 0 else float('-inf'))


class MCTSStrategy(BaseStrategy):
    """Monte Carlo Tree Search strategy."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize MCTS strategy."""
        super().__init__(config)
        self.exploration_constant = config.get('exploration_constant', 1.4)
        self.simulation_count = config.get('simulation_count', 100)

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using MCTS."""
        operations = [
            ('xor_constant', {'constant': random.randint(1, 255)}),
            ('rotate_left', {'shift': random.randint(1, 7)}),
            ('move_to_front', {}),
            ('shuffle_bytes', {'seed': random.randint(0, 10000)})
        ]
        return random.choice(operations)

    def accept(self, new_state: State) -> bool:
        """Accept based on MCTS evaluation."""
        return new_state.score > self.best_score