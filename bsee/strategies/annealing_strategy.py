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
    """Simulated annealing strategy with temperature-based operation selection."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize annealing strategy."""
        super().__init__(config)
        self.load_config()
        self.current_temperature = self.initial_temperature
        self.operations_registry = None
        self.acceptance_history = []  # Track acceptance rate for adaptive cooling
        self.reheat_count = 0
        self.last_improvement_iteration = 0

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_path = Path("config/strategies/strategy_annealing.yaml")

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Load temperature parameters
            temp = yaml_config.get('temperature', {})
            self.initial_temperature = temp.get('initial_temperature', 100.0)
            self.min_temperature = temp.get('min_temperature', 0.1)
            self.cooling_schedule = temp.get('cooling_schedule', 'geometric')
            self.cooling_rate = temp.get('cooling_rate', 0.95)
            self.linear_decrement = temp.get('linear_decrement', 0.5)
            self.exponential_alpha = temp.get('exponential_alpha', 0.99)
            self.adaptive_cooling = temp.get('adaptive_cooling', True)

            # Load acceptance parameters
            acceptance = yaml_config.get('acceptance', {})
            self.always_accept_improvements = acceptance.get('always_accept_improvements', True)
            self.use_boltzmann = acceptance.get('use_boltzmann', True)
            self.accept_equal_probability = acceptance.get('accept_equal_probability', 0.1)

            # Load restart parameters
            restart = yaml_config.get('restart_strategy', {})
            self.enable_reheating = restart.get('enable_reheating', True)
            self.reheat_threshold = restart.get('reheat_threshold', 50)
            self.reheat_multiplier = restart.get('reheat_multiplier', 0.5)
            self.max_reheats = restart.get('max_reheats', 5)

            # Load adaptive parameters
            adaptive = yaml_config.get('adaptive', {})
            self.adaptive_cooling_rate = adaptive.get('adaptive_cooling_rate', True)
            self.performance_window = adaptive.get('performance_window', 20)
            self.target_acceptance_rate = adaptive.get('target_acceptance_rate', 0.4)
            self.cooling_adjustment = adaptive.get('cooling_adjustment', 0.1)

            # Load exploration parameters
            exploration = yaml_config.get('exploration', {})
            self.high_temp_operations = exploration.get('high_temp_operations',
                                                     ["shuffle_bytes", "reverse_bytes", "base64_encode"])
            self.low_temp_operations = exploration.get('low_temp_operations',
                                                    ["xor_constant", "toggle_bit", "clear_bit"])
            self.high_temp_threshold = exploration.get('high_temp_threshold', 50.0)
            self.low_temp_threshold = exploration.get('low_temp_threshold', 5.0)

            # Merge with existing config dict
            self.config.update(yaml_config)

        except FileNotFoundError:
            # Use default values if config file missing
            self.initial_temperature = 100.0
            self.min_temperature = 0.1
            self.cooling_schedule = 'geometric'
            self.cooling_rate = 0.95
            self.linear_decrement = 0.5
            self.exponential_alpha = 0.99
            self.adaptive_cooling = True
            self.always_accept_improvements = True
            self.use_boltzmann = True
            self.accept_equal_probability = 0.1
            self.enable_reheating = True
            self.reheat_threshold = 50
            self.reheat_multiplier = 0.5
            self.max_reheats = 5
            self.adaptive_cooling_rate = True
            self.performance_window = 20
            self.target_acceptance_rate = 0.4
            self.cooling_adjustment = 0.1
            self.high_temp_operations = ["shuffle_bytes", "reverse_bytes", "base64_encode"]
            self.low_temp_operations = ["xor_constant", "toggle_bit", "clear_bit"]
            self.high_temp_threshold = 50.0
            self.low_temp_threshold = 5.0
        except yaml.YAMLError as e:
            # Log error and use defaults
            print(f"Warning: Error loading annealing config: {e}")
            self.initial_temperature = 100.0
            self.cooling_rate = 0.95
            self.min_temperature = 0.1

    def set_operations_registry(self, operations_registry) -> None:
        """Set the operations registry for accessing available operations."""
        self.operations_registry = operations_registry

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using temperature-based selection."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Select operation based on temperature
            operation_name = self._select_temperature_based_operation()
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = self._generate_operation_params(metadata)

            return (operation_name, params)

        except Exception as e:
            print(f"Warning: Annealing strategy failed, using fallback: {e}")
            return self._fallback_proposal(current_state)

    def _select_temperature_based_operation(self) -> str:
        """Select operation based on current temperature."""
        available_operations = self.operations_registry.list_operations()

        # Categorize operations by temperature
        high_temp_available = [op for op in self.high_temp_operations if op in available_operations]
        low_temp_available = [op for op in self.low_temp_operations if op in available_operations]
        other_operations = [op for op in available_operations
                           if op not in high_temp_operations and op not in low_temp_operations]

        # Determine operation selection based on temperature
        if self.current_temperature > self.high_temp_threshold:
            # High temperature: prefer exploratory operations
            if high_temp_available:
                return random.choice(high_temp_available)
            else:
                return random.choice(available_operations)
        elif self.current_temperature > self.low_temp_threshold:
            # Medium temperature: mix of exploration and exploitation
            if random.random() < self.current_temperature / self.high_temp_threshold:
                # More likely to choose exploratory
                candidates = high_temp_available if high_temp_available else other_operations
            else:
                # More likely to choose exploitative
                candidates = low_temp_available if low_temp_available else other_operations

            if candidates:
                return random.choice(candidates)
            else:
                return random.choice(available_operations)
        else:
            # Low temperature: prefer exploitative operations
            if low_temp_available:
                return random.choice(low_temp_available)
            else:
                return random.choice(available_operations)

    def accept(self, new_state: State) -> bool:
        """Accept state based on simulated annealing criteria with proper Boltzmann acceptance."""
        current_score = self.best_score
        new_score = new_state.score

        # Always accept improvements if configured
        if self.always_accept_improvements and new_score > current_score:
            self.last_improvement_iteration = self.iteration_count
            self.acceptance_history.append(True)
            self._cool_temperature()
            return True

        # Calculate acceptance probability
        acceptance_probability = self._calculate_acceptance_probability(current_score, new_score)

        # Accept or reject based on probability
        accepted = random.random() < acceptance_probability

        if accepted:
            if new_score > current_score:
                self.last_improvement_iteration = self.iteration_count
            self.acceptance_history.append(True)
        else:
            self.acceptance_history.append(False)

        # Cool down temperature
        self._cool_temperature()

        # Check for reheating
        self._check_reheating()

        # Trim acceptance history
        if len(self.acceptance_history) > self.performance_window:
            self.acceptance_history = self.acceptance_history[-self.performance_window:]

        return accepted

    def _calculate_acceptance_probability(self, current_score: float, new_score: float) -> float:
        """Calculate acceptance probability using Boltzmann distribution."""
        if self.use_boltzmann:
            # Boltzmann acceptance: exp(delta_score / temperature)
            delta_score = new_score - current_score
            if self.current_temperature <= 0:
                return 0.0

            # Handle edge cases
            if delta_score > 0:
                # Improvement: always accept (or very high probability)
                return 1.0 if self.always_accept_improvements else min(1.0, math.exp(delta_score / self.current_temperature))
            elif delta_score == 0:
                # Equal score: use configured probability
                return self.accept_equal_probability
            else:
                # Worse score: accept with probability based on temperature
                try:
                    probability = math.exp(delta_score / self.current_temperature)
                    return max(0.0, min(1.0, probability))
                except OverflowError:
                    return 0.0
        else:
            # Simple threshold-based acceptance
            if new_score >= current_score:
                return 1.0
            else:
                # Linear scaling based on temperature
                temp_ratio = self.current_temperature / self.initial_temperature
                return max(0.0, temp_ratio)

    def _cool_temperature(self) -> None:
        """Cool down temperature according to the cooling schedule."""
        if self.current_temperature <= self.min_temperature:
            return

        if self.adaptive_cooling and len(self.acceptance_history) >= self.performance_window:
            # Adaptive cooling based on acceptance rate
            acceptance_rate = sum(self.acceptance_history) / len(self.acceptance_history)

            if acceptance_rate > self.target_acceptance_rate:
                # Accepting too much, cool faster
                adjusted_cooling_rate = self.cooling_rate * (1 + self.cooling_adjustment)
            elif acceptance_rate < self.target_acceptance_rate * 0.5:
                # Accepting too little, cool slower
                adjusted_cooling_rate = self.cooling_rate * (1 - self.cooling_adjustment)
            else:
                # Good acceptance rate, use normal cooling
                adjusted_cooling_rate = self.cooling_rate

            adjusted_cooling_rate = max(0.8, min(0.99, adjusted_cooling_rate))  # Keep reasonable bounds
        else:
            adjusted_cooling_rate = self.cooling_rate

        # Apply cooling based on schedule
        if self.cooling_schedule == 'geometric':
            self.current_temperature *= adjusted_cooling_rate
        elif self.cooling_schedule == 'linear':
            self.current_temperature -= self.linear_decrement
        elif self.cooling_schedule == 'exponential':
            self.current_temperature = self.initial_temperature * (self.exponential_alpha ** self.iteration_count)
        else:
            # Default to geometric
            self.current_temperature *= adjusted_cooling_rate

        # Ensure temperature doesn't go below minimum
        self.current_temperature = max(self.current_temperature, self.min_temperature)

    def _check_reheating(self) -> None:
        """Check if reheating is needed and perform it if conditions are met."""
        if not self.enable_reheating:
            return

        if self.reheat_count >= self.max_reheats:
            return

        # Check if stuck (no improvement for threshold iterations)
        iterations_since_improvement = self.iteration_count - self.last_improvement_iteration
        if iterations_since_improvement >= self.reheat_threshold:
            # Reheat
            reheated_temperature = self.initial_temperature * self.reheat_multiplier
            self.current_temperature = max(self.current_temperature, reheated_temperature)
            self.reheat_count += 1
            print(f"Reheating to {self.current_temperature:.2f} (reheat #{self.reheat_count})")

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
                    # Temperature affects parameter selection
                    if self.current_temperature > self.high_temp_threshold:
                        # High temperature: more extreme values
                        range_ext = (max_val - min_val) * 0.2
                        params[param_name] = random.randint(
                            max(min_val, min_val - range_ext),
                            max(max_val, max_val + range_ext)
                        )
                    else:
                        params[param_name] = random.randint(min_val, max_val)
                elif param_type == 'float':
                    min_val = param_info.get('min', 0.0)
                    max_val = param_info.get('max', 1.0)
                    params[param_name] = random.uniform(min_val, max_val)
                elif param_type == 'bool':
                    params[param_name] = random.choice([True, False])

        return params

    def _fallback_proposal(self, current_state: Optional[State]) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal when annealing strategy fails."""
        if self.operations_registry:
            operations = self.operations_registry.list_operations()
            operation_name = random.choice(operations)
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = self._generate_operation_params(metadata)
            return (operation_name, params)
        else:
            # Hardcoded fallback
            return ('xor_constant', {'constant': random.randint(1, 255)})

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.current_temperature = self.initial_temperature
        self.acceptance_history.clear()
        self.reheat_count = 0
        self.last_improvement_iteration = 0

    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about the strategy's current state."""
        base_info = super().get_strategy_info()
        base_info.update({
            'current_temperature': self.current_temperature,
            'reheat_count': self.reheat_count,
            'acceptance_rate': (sum(self.acceptance_history) / len(self.acceptance_history)
                              if self.acceptance_history else 0.0)
        })
        return base_info