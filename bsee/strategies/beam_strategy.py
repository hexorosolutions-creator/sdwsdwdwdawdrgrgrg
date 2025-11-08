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
        self.load_config()
        self.beam: List[BeamState] = []
        self.operations_registry = None
        self.operation_history: Dict[str, float] = {}  # Track operation performance

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_path = Path("config/strategies/strategy_beam.yaml")

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Load parameters from YAML with defaults
            params = yaml_config.get('parameters', {})
            self.beam_width = params.get('beam_width', 5)
            self.max_beam_width = params.get('max_beam_width', 20)
            self.min_beam_width = params.get('min_beam_width', 2)
            self.branch_factor = params.get('branch_factor', 3)

            # Load beam management parameters
            beam_mgmt = yaml_config.get('beam_management', {})
            self.selection_method = beam_mgmt.get('selection_method', 'top_n')
            self.preserve_diversity = beam_mgmt.get('preserve_diversity', True)
            self.diversity_threshold = beam_mgmt.get('diversity_threshold', 0.1)
            self.normalize_scores = beam_mgmt.get('normalize_scores', True)
            self.aging_factor = beam_mgmt.get('aging_factor', 0.99)

            # Load dynamic beam parameters
            dynamic_beam = yaml_config.get('dynamic_beam', {})
            self.dynamic_beam_enabled = dynamic_beam.get('enabled', True)
            self.expansion_threshold = dynamic_beam.get('expansion_threshold', 0.8)
            self.contraction_threshold = dynamic_beam.get('contraction_threshold', 0.2)
            self.adjustment_rate = dynamic_beam.get('adjustment_rate', 0.1)

            # Load operation selection parameters
            op_selection = yaml_config.get('operation_selection', {})
            self.diverse_operations = op_selection.get('diverse_operations', True)
            self.min_operation_diversity = op_selection.get('min_operation_diversity', 0.5)
            self.use_historical_performance = op_selection.get('use_historical_performance', True)

            # Load termination parameters
            termination = yaml_config.get('termination', {})
            self.score_similarity_threshold = termination.get('score_similarity_threshold', 0.01)
            self.state_similarity_threshold = termination.get('state_similarity_threshold', 0.9)

            # Merge with existing config dict
            self.config.update(yaml_config)

        except FileNotFoundError:
            # Use default values if config file missing
            self.beam_width = 5
            self.max_beam_width = 20
            self.min_beam_width = 2
            self.branch_factor = 3
            self.selection_method = 'top_n'
            self.preserve_diversity = True
            self.diversity_threshold = 0.1
            self.normalize_scores = True
            self.aging_factor = 0.99
            self.dynamic_beam_enabled = True
            self.expansion_threshold = 0.8
            self.contraction_threshold = 0.2
            self.adjustment_rate = 0.1
            self.diverse_operations = True
            self.min_operation_diversity = 0.5
            self.use_historical_performance = True
            self.score_similarity_threshold = 0.01
            self.state_similarity_threshold = 0.9
        except yaml.YAMLError as e:
            # Log error and use defaults
            print(f"Warning: Error loading beam config: {e}")
            self.beam_width = 5
            self.branch_factor = 3

    def set_operations_registry(self, operations_registry) -> None:
        """Set the operations registry for accessing available operations."""
        self.operations_registry = operations_registry

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using beam search."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Initialize beam if needed
            if not self.beam:
                self.initialize_beam(current_state)

            # Expand beam with new candidates
            expanded_states = self.expand_beam()

            # Prune beam to keep best candidates
            if expanded_states:
                self.prune_beam(expanded_states)

            # Adjust beam width dynamically if enabled
            if self.dynamic_beam_enabled:
                self.adjust_beam_width()

            # Check convergence
            if self._check_convergence():
                self.converged = True

            # Select best operation from best beam state
            return self.select_best_operation()

        except Exception as e:
            print(f"Warning: Beam strategy failed, using fallback: {e}")
            return self._fallback_proposal(current_state)

    def initialize_beam(self, current_state: State) -> None:
        """Initialize beam with random operations."""
        self.beam = []
        available_operations = self.operations_registry.list_operations()

        # Create initial beam states with different single operations
        for i in range(self.beam_width):
            try:
                # Select operation for diversity
                if self.diverse_operations and i < len(available_operations):
                    operation_name = available_operations[i]
                else:
                    operation_name = random.choice(available_operations)

                metadata = self.operations_registry.get_operation_metadata(operation_name)
                params = self._generate_operation_params(metadata)

                # Apply operation
                operation_func = self.operations_registry.get_operation(operation_name)
                new_data, _, _ = operation_func(current_state.data, **params)
                new_state = State(new_data, self.operations_registry)

                # Create beam state
                beam_state = BeamState(
                    new_state,
                    [(operation_name, params)],
                    new_state.score
                )
                self.beam.append(beam_state)

            except Exception as e:
                print(f"Warning: Failed to create beam state {i}: {e}")
                continue

        # Ensure we have at least one beam state
        if not self.beam:
            # Fallback: add current state
            beam_state = BeamState(current_state, [], current_state.score)
            self.beam.append(beam_state)

    def expand_beam(self) -> List[BeamState]:
        """Expand each beam state with possible operations."""
        expanded_states = []
        available_operations = self.operations_registry.list_operations()

        for beam_state in self.beam:
            # Get diverse operations based on history
            operations_to_try = self._select_diverse_operations(beam_state, available_operations)

            for operation_name in operations_to_try[:self.branch_factor]:
                try:
                    # Generate parameters
                    metadata = self.operations_registry.get_operation_metadata(operation_name)
                    params = self._generate_operation_params(metadata)

                    # Apply operation
                    operation_func = self.operations_registry.get_operation(operation_name)
                    new_data, _, _ = operation_func(beam_state.state.data, **params)
                    new_state = State(new_data, self.operations_registry)

                    # Create new beam state
                    new_beam_state = BeamState(
                        new_state,
                        beam_state.operations_applied + [(operation_name, params)],
                        new_state.score
                    )
                    expanded_states.append(new_beam_state)

                    # Update operation performance history
                    self._update_operation_history(operation_name, new_state.score)

                except Exception as e:
                    # Skip operations that fail
                    continue

        return expanded_states

    def _select_diverse_operations(self, beam_state: BeamState, available_operations: List[str]) -> List[str]:
        """Select diverse operations for beam expansion."""
        if not self.diverse_operations:
            return random.sample(available_operations, min(self.branch_factor, len(available_operations)))

        # Get operations already used in this beam state
        used_operations = {op[0] for op in beam_state.operations_applied}

        # Prioritize unused operations
        unused_operations = [op for op in available_operations if op not in used_operations]

        if len(unused_operations) >= self.branch_factor:
            selected = unused_operations[:self.branch_factor]
        else:
            # Use all unused and add some used ones
            selected = unused_operations
            remaining = self.branch_factor - len(selected)
            if remaining > 0:
                used_ops_list = [op for op in available_operations if op in used_operations]
                selected.extend(random.sample(used_ops_list, min(remaining, len(used_ops_list))))

        # If using historical performance, sort by performance
        if self.use_historical_performance and self.operation_history:
            selected.sort(key=lambda op: self.operation_history.get(op, 0.0), reverse=True)

        return selected

    def prune_beam(self, expanded_states: List[BeamState]) -> None:
        """Prune beam to keep best candidates."""
        # Combine current beam and expanded states
        all_states = self.beam + expanded_states

        # Age all states
        for state in all_states:
            state.age += 1
            # Apply aging factor to score
            state.score *= self.aging_factor

        # Calculate diversity scores if diversity preservation is enabled
        if self.preserve_diversity:
            for state in all_states:
                state.diversity_score = state.calculate_diversity(all_states)

        # Normalize scores if enabled
        if self.normalize_scores and all_states:
            min_score = min(state.score for state in all_states)
            max_score = max(state.score for state in all_states)
            score_range = max_score - min_score

            if score_range > 0:
                for state in all_states:
                    state.normalized_score = (state.score - min_score) / score_range
            else:
                for state in all_states:
                    state.normalized_score = 0.5

        # Select states based on selection method
        if self.selection_method == 'top_n':
            selected_states = self._select_top_n(all_states)
        elif self.selection_method == 'diverse':
            selected_states = self._select_diverse(all_states)
        elif self.selection_method == 'weighted':
            selected_states = self._select_weighted(all_states)
        else:
            selected_states = self._select_top_n(all_states)

        # Update beam
        self.beam = selected_states[:self.beam_width]

    def _select_top_n(self, states: List[BeamState]) -> List[BeamState]:
        """Select top N states by score."""
        return sorted(states, key=lambda state: state.score, reverse=True)

    def _select_diverse(self, states: List[BeamState]) -> List[BeamState]:
        """Select diverse states maintaining high scores."""
        if not states:
            return []

        # Sort by score
        sorted_states = sorted(states, key=lambda state: state.score, reverse=True)
        selected = [sorted_states[0]]  # Always include the best state

        for candidate in sorted_states[1:]:
            # Calculate diversity against selected states
            candidate.diversity_score = candidate.calculate_diversity(selected)

            # Select if diverse enough or if we haven't filled beam
            if (candidate.diversity_score >= self.diversity_threshold or
                len(selected) < self.beam_width):
                selected.append(candidate)

            if len(selected) >= self.beam_width:
                break

        return selected

    def _select_weighted(self, states: List[BeamState]) -> List[BeamState]:
        """Select states using weighted combination of score and diversity."""
        if not states:
            return []

        # Calculate combined scores
        for state in states:
            if self.preserve_diversity:
                state.diversity_score = state.calculate_diversity(states)
                state.combined_score = (0.7 * state.score +
                                      0.3 * state.diversity_score)
            else:
                state.combined_score = state.score

        # Sort by combined score
        return sorted(states, key=lambda state: state.combined_score, reverse=True)

    def adjust_beam_width(self) -> None:
        """Dynamically adjust beam width based on performance."""
        if not self.beam or len(self.beam) < 2:
            return

        # Calculate improvement rate
        scores = [state.score for state in self.beam]
        score_variance = max(scores) - min(scores)

        # Adjust beam width based on improvement
        if score_variance > self.expansion_threshold:
            # Many improvements - expand beam
            new_width = min(self.beam_width + 1, self.max_beam_width)
            self.beam_width = int(self.beam_width + (new_width - self.beam_width) * self.adjustment_rate)
        elif score_variance < self.contraction_threshold:
            # Stuck - contract beam
            new_width = max(self.beam_width - 1, self.min_beam_width)
            self.beam_width = int(self.beam_width - (self.beam_width - new_width) * self.adjustment_rate)

    def _check_convergence(self) -> bool:
        """Check if the beam has converged."""
        if not self.beam or len(self.beam) < 2:
            return False

        # Check score similarity
        scores = [state.score for state in self.beam]
        score_range = max(scores) - min(scores)
        if score_range < self.score_similarity_threshold:
            return True

        # Check state similarity (operation sequences)
        if len(self.beam) > 1:
            reference_ops = set(self.beam[0].operations_applied)
            for state in self.beam[1:]:
                current_ops = set(state.operations_applied)
                if not current_ops:  # Empty operation sequence
                    continue

                # Calculate Jaccard similarity
                intersection = len(reference_ops.intersection(current_ops))
                union = len(reference_ops.union(current_ops))
                similarity = intersection / union if union > 0 else 0

                if similarity > self.state_similarity_threshold:
                    return True

        return False

    def select_best_operation(self) -> Tuple[str, Dict[str, Any]]:
        """Select the best operation from the beam."""
        if not self.beam:
            return self._fallback_proposal(None)

        # Get best beam state
        best_state = max(self.beam, key=lambda state: state.score)

        if best_state.operations_applied:
            # Return the last operation applied to the best state
            return best_state.operations_applied[-1]
        else:
            # No operations applied yet, fallback
            return self._fallback_proposal(best_state.state)

    def _update_operation_history(self, operation_name: str, score: float) -> None:
        """Update historical performance of operations."""
        if operation_name not in self.operation_history:
            self.operation_history[operation_name] = 0.0

        # Exponential moving average
        alpha = 0.1
        self.operation_history[operation_name] = (
            alpha * score + (1 - alpha) * self.operation_history[operation_name]
        )

    def _generate_operation_params(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate valid parameters for an operation."""
        params = {}
        param_defs = metadata.get('parameters', {})

        for param_name, param_info in param_defs.items():
            if param_info.get('required', False):
                param_type = param_info.get('type', 'auto')
                default = param_info.get('default')

                if default is not None:
                    params[param_name] = default
                elif param_type == 'int':
                    min_val = param_info.get('min', 1)
                    max_val = param_info.get('max', 255)
                    params[param_name] = random.randint(min_val, max_val)
                elif param_type == 'float':
                    min_val = param_info.get('min', 0.0)
                    max_val = param_info.get('max', 1.0)
                    params[param_name] = random.uniform(min_val, max_val)
                elif param_type == 'bool':
                    params[param_name] = random.choice([True, False])

        return params

    def _fallback_proposal(self, current_state: Optional[State]) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal when beam search fails."""
        if self.operations_registry:
            operations = self.operations_registry.list_operations()
            operation_name = random.choice(operations)
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = self._generate_operation_params(metadata)
            return (operation_name, params)
        else:
            # Hardcoded fallback
            return ('xor_constant', {'constant': random.randint(1, 255)})

    def accept(self, new_state: State) -> bool:
        """Accept state based on beam criteria."""
        # Update best score
        if new_state.score > self.best_score:
            self.best_score = new_state.score
            self.no_improvement_count = 0
            return True
        else:
            self.no_improvement_count += 1
            return False

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.beam = []
        self.operation_history.clear()