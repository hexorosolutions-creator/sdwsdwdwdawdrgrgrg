"""
Base strategy interface for BSEE search strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple
from bsee.engine.state import State
from bsee.utils.error_handler import handle_exception, ErrorContext


class BaseStrategy(ABC):
    """Abstract base class for search strategies."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy with configuration."""
        self.config = config
        self.name = self.__class__.__name__
        self.iteration_count = 0
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.converged = False

    @abstractmethod
    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose next operation to apply.

        Args:
            current_state: Current state of the binary data

        Returns:
            Tuple of (operation_name, operation_parameters)
        """
        pass

    @abstractmethod
    def accept(self, new_state: State) -> bool:
        """Decide whether to accept a new state.

        Args:
            new_state: Proposed new state

        Returns:
            True if the state should be accepted, False otherwise
        """
        pass

    def is_converged(self) -> bool:
        """Check if the search has converged."""
        return self.converged

    def reset(self) -> None:
        """Reset the strategy state."""
        self.iteration_count = 0
        self.best_score = float('-inf')
        self.no_improvement_count = 0
        self.converged = False

    def update_statistics(self, state: State, accepted: bool) -> None:
        """Update strategy statistics based on state evaluation."""
        self.iteration_count += 1

        if accepted and state.score > self.best_score:
            self.best_score = state.score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Check convergence criteria
        self._check_convergence()

    def _check_convergence(self) -> None:
        """Check if convergence criteria are met."""
        max_no_improvement = self.config.get('max_no_improvement', 50)
        max_iterations = self.config.get('max_iterations', 1000)

        if self.no_improvement_count >= max_no_improvement:
            self.converged = True

        if self.iteration_count >= max_iterations:
            self.converged = True

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy's current state."""
        return {
            'name': self.name,
            'iteration_count': self.iteration_count,
            'best_score': self.best_score,
            'no_improvement_count': self.no_improvement_count,
            'converged': self.converged
        }