"""
Heuristic strategy for BSEE.
"""

import random
import math
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from collections import defaultdict, deque
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class HeuristicStrategy(BaseStrategy):
    """Multi-criteria heuristic strategy with pattern recognition."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize heuristic strategy."""
        super().__init__(config)
        self.load_config()
        self.operations_registry = None
        self.operation_history = deque(maxlen=self.heuristic_window)
        self.operation_performance = defaultdict(list)
        self.pattern_memory = {}
        self.data_analyzer = None

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_path = Path("config/strategies/strategy_heuristic.yaml")

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Load parameters from YAML with defaults
            params = yaml_config.get('parameters', {})
            self.heuristic_window = params.get('heuristic_window', 50)
            self.diversity_threshold = params.get('diversity_threshold', 0.3)
            self.pattern_sensitivity = params.get('pattern_sensitivity', 0.7)

            # Load heuristic weights
            heuristics = yaml_config.get('heuristics', {})
            self.cost_weight = heuristics.get('cost_weight', 0.2)
            self.improvement_weight = heuristics.get('improvement_weight', 0.4)
            self.diversity_weight = heuristics.get('diversity_weight', 0.2)
            self.pattern_weight = heuristics.get('pattern_weight', 0.2)

            # Load pattern recognition parameters
            patterns = yaml_config.get('pattern_recognition', {})
            self.enable_pattern_recognition = patterns.get('enabled', True)
            self.pattern_min_occurrences = patterns.get('min_occurrences', 3)
            self.pattern_confidence_threshold = patterns.get('confidence_threshold', 0.8)

            # Load adaptive learning parameters
            learning = yaml_config.get('adaptive_learning', {})
            self.enable_adaptive_learning = learning.get('enabled', True)
            self.learning_rate = learning.get('learning_rate', 0.1)
            self.performance_decay = learning.get('performance_decay', 0.95)

            # Merge with existing config dict
            self.config.update(yaml_config)

        except FileNotFoundError:
            # Create default config file
            self._create_default_config()
            # Use default values
            self.heuristic_window = 50
            self.diversity_threshold = 0.3
            self.pattern_sensitivity = 0.7
            self.cost_weight = 0.2
            self.improvement_weight = 0.4
            self.diversity_weight = 0.2
            self.pattern_weight = 0.2
            self.enable_pattern_recognition = True
            self.pattern_min_occurrences = 3
            self.pattern_confidence_threshold = 0.8
            self.enable_adaptive_learning = True
            self.learning_rate = 0.1
            self.performance_decay = 0.95
        except yaml.YAMLError as e:
            # Log error and use defaults
            print(f"Warning: Error loading heuristic config: {e}")
            self.heuristic_window = 50
            self.cost_weight = 0.2
            self.improvement_weight = 0.4
            self.diversity_weight = 0.2
            self.pattern_weight = 0.2

    def _create_default_config(self) -> None:
        """Create default heuristic strategy config file."""
        config_dir = Path("config/strategies")
        config_dir.mkdir(parents=True, exist_ok=True)

        default_config = {
            'name': 'Heuristic Strategy',
            'description': 'Multi-criteria heuristic evaluation with pattern recognition',
            'parameters': {
                'heuristic_window': 50,
                'diversity_threshold': 0.3,
                'pattern_sensitivity': 0.7,
                'max_patterns': 100
            },
            'heuristics': {
                'cost_weight': 0.2,        # Computational cost consideration
                'improvement_weight': 0.4,  # Expected score improvement
                'diversity_weight': 0.2,   # Operation diversity promotion
                'pattern_weight': 0.2      # Pattern-based selection
            },
            'pattern_recognition': {
                'enabled': True,
                'min_occurrences': 3,
                'confidence_threshold': 0.8,
                'max_pattern_length': 5
            },
            'adaptive_learning': {
                'enabled': True,
                'learning_rate': 0.1,
                'performance_decay': 0.95,
                'min_samples': 5
            },
            'data_analysis': {
                'entropy_threshold': 0.8,
                'repetition_threshold': 0.6,
                'complexity_threshold': 0.5
            }
        }

        config_path = config_dir / "strategy_heuristic.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def set_operations_registry(self, operations_registry) -> None:
        """Set the operations registry for accessing available operations."""
        self.operations_registry = operations_registry

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using multi-criteria heuristic evaluation."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Analyze current state data patterns
            data_features = self._analyze_data_patterns(current_state)

            # Get available operations
            available_operations = self.operations_registry.list_operations()

            # Evaluate each operation using multiple heuristics
            operation_scores = []
            for operation_name in available_operations:
                score = self._evaluate_operation_heuristic(operation_name, current_state, data_features)
                operation_scores.append((operation_name, score))

            # Select best operation based on combined heuristic score
            if operation_scores:
                operation_scores.sort(key=lambda x: x[1], reverse=True)
                best_operation = operation_scores[0][0]
                metadata = self.operations_registry.get_operation_metadata(best_operation)
                params = self._generate_operation_params(metadata, data_features)

                # Add to operation history
                self.operation_history.append((best_operation, params, current_state.score))

                return (best_operation, params)
            else:
                return self._fallback_proposal(current_state)

        except Exception as e:
            print(f"Warning: Heuristic strategy failed, using fallback: {e}")
            return self._fallback_proposal(current_state)

    def _analyze_data_patterns(self, state: State) -> Dict[str, float]:
        """Analyze patterns in the current state data."""
        data = state.data
        if not data:
            return {}

        features = {}

        # Calculate basic statistics
        if len(data) > 0:
            # Byte frequency analysis
            byte_counts = defaultdict(int)
            for byte in data:
                byte_counts[byte] += 1

            # Entropy calculation
            total_bytes = len(data)
            entropy = 0.0
            for count in byte_counts.values():
                if count > 0:
                    probability = count / total_bytes
                    entropy -= probability * math.log2(probability)
            features['entropy'] = entropy / 8.0  # Normalize to [0, 1]

            # Repetition detection
            repeated_bytes = sum(1 for count in byte_counts.values() if count > 1)
            features['repetition_ratio'] = repeated_bytes / 256.0

            # Pattern detection (simple runs)
            max_run_length = 1
            current_run = 1
            for i in range(1, len(data)):
                if data[i] == data[i-1]:
                    current_run += 1
                    max_run_length = max(max_run_length, current_run)
                else:
                    current_run = 1
            features['max_run_length'] = min(1.0, max_run_length / len(data))

            # Data complexity (adjacent byte differences)
            if len(data) > 1:
                differences = sum(1 for i in range(1, len(data)) if data[i] != data[i-1])
                features['complexity'] = differences / (len(data) - 1)
            else:
                features['complexity'] = 0.0

        return features

    def _evaluate_operation_heuristic(self, operation_name: str, current_state: State,
                                    data_features: Dict[str, float]) -> float:
        """Evaluate operation using multiple heuristic criteria."""
        # Get operation metadata
        metadata = self.operations_registry.get_operation_metadata(operation_name)

        # Cost heuristic - prefer computationally cheaper operations
        cost_score = self._calculate_cost_heuristic(metadata)

        # Improvement heuristic - expected score improvement based on historical performance
        improvement_score = self._calculate_improvement_heuristic(operation_name, data_features)

        # Diversity heuristic - promote operations different from recent history
        diversity_score = self._calculate_diversity_heuristic(operation_name)

        # Pattern heuristic - operations effective on current data patterns
        pattern_score = self._calculate_pattern_heuristic(operation_name, data_features)

        # Combine heuristics using weighted sum
        total_score = (
            self.cost_weight * cost_score +
            self.improvement_weight * improvement_score +
            self.diversity_weight * diversity_score +
            self.pattern_weight * pattern_score
        )

        return total_score

    def _calculate_cost_heuristic(self, metadata: Dict[str, Any]) -> float:
        """Calculate cost heuristic - lower cost is better."""
        # Get complexity from metadata, default to medium
        complexity = metadata.get('complexity', 'medium')

        # Map complexity to cost score (lower cost = higher score)
        cost_scores = {
            'low': 1.0,
            'medium': 0.7,
            'high': 0.3
        }

        return cost_scores.get(complexity, 0.7)

    def _calculate_improvement_heuristic(self, operation_name: str,
                                       data_features: Dict[str, float]) -> float:
        """Calculate improvement heuristic based on historical performance."""
        if not self.enable_adaptive_learning:
            return 0.5  # Default neutral score

        # Get historical performance data
        performance_history = self.operation_performance.get(operation_name, [])

        if not performance_history:
            return 0.5  # No history, neutral score

        # Calculate average recent performance with decay
        recent_performances = performance_history[-10:]  # Last 10 uses
        if recent_performances:
            # Apply exponential decay to older performances
            weighted_sum = 0.0
            total_weight = 0.0

            for i, score in enumerate(reversed(recent_performances)):
                weight = (self.performance_decay ** i)
                weighted_sum += score * weight
                total_weight += weight

            average_performance = weighted_sum / total_weight if total_weight > 0 else 0.5
        else:
            average_performance = 0.5

        # Adjust based on data features
        entropy = data_features.get('entropy', 0.5)
        complexity = data_features.get('complexity', 0.5)

        # Some operations work better on different data types
        operation_category = self._get_operation_category(operation_name)

        if operation_category == 'transform':
            # Transform operations often work well on high-entropy data
            entropy_bonus = entropy * 0.2
        elif operation_category == 'reordering':
            # Reordering operations work well on repetitive data
            repetition_bonus = (1.0 - entropy) * 0.2
            entropy_bonus = repetition_bonus
        else:
            entropy_bonus = 0.0

        return min(1.0, average_performance + entropy_bonus)

    def _calculate_diversity_heuristic(self, operation_name: str) -> float:
        """Calculate diversity heuristic to promote operation variety."""
        if not self.operation_history:
            return 1.0  # No history, full diversity

        # Count recent uses of this operation
        recent_uses = sum(1 for op, _, _ in self.operation_history if op == operation_name)

        # Calculate diversity score (lower recent usage = higher diversity score)
        if recent_uses == 0:
            return 1.0

        # Exponential decay based on recent usage
        diversity_score = math.exp(-recent_uses / self.diversity_threshold)
        return diversity_score

    def _calculate_pattern_heuristic(self, operation_name: str,
                                   data_features: Dict[str, float]) -> float:
        """Calculate pattern heuristic based on data pattern recognition."""
        if not self.enable_pattern_recognition:
            return 0.5  # Disabled, neutral score

        # Create pattern key from current data features
        pattern_key = self._create_pattern_key(data_features)

        # Check if we have learned patterns for this situation
        if pattern_key in self.pattern_memory:
            pattern_data = self.pattern_memory[pattern_key]
            operation_effectiveness = pattern_data.get(operation_name, 0.5)

            # Consider confidence in the pattern
            confidence = pattern_data.get('confidence', 0.5)
            occurrences = pattern_data.get('occurrences', 1)

            if occurrences >= self.pattern_min_occurrences and confidence >= self.pattern_confidence_threshold:
                return operation_effectiveness

        # Use rule-based heuristic for unknown patterns
        return self._rule_based_pattern_heuristic(operation_name, data_features)

    def _rule_based_pattern_heuristic(self, operation_name: str,
                                    data_features: Dict[str, float]) -> float:
        """Rule-based heuristic when no learned patterns are available."""
        entropy = data_features.get('entropy', 0.5)
        repetition = data_features.get('repetition_ratio', 0.5)
        complexity = data_features.get('complexity', 0.5)

        operation_category = self._get_operation_category(operation_name)

        # Rule-based scoring based on data characteristics
        if operation_category == 'transform':
            # Transform operations for high entropy/complexity data
            return (entropy + complexity) / 2.0
        elif operation_category == 'reordering':
            # Reordering for repetitive data
            return (1.0 - entropy) * 0.7 + complexity * 0.3
        elif operation_category == 'bitwise':
            # Bitwise operations for moderate entropy
            return 0.8 - abs(entropy - 0.5)
        else:
            return 0.5  # Default for unknown categories

    def _get_operation_category(self, operation_name: str) -> str:
        """Get the category of an operation."""
        try:
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            return metadata.get('category', 'unknown')
        except:
            return 'unknown'

    def _create_pattern_key(self, data_features: Dict[str, float]) -> str:
        """Create a pattern key from data features."""
        # Discretize features to create pattern classes
        entropy_level = 'high' if data_features.get('entropy', 0) > 0.7 else 'low'
        complexity_level = 'high' if data_features.get('complexity', 0) > 0.6 else 'low'

        return f"{entropy_level}_entropy_{complexity_level}_complexity"

    def _generate_operation_params(self, metadata: Dict[str, Any],
                                 data_features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate parameters for an operation based on data features."""
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

                    # Adapt parameters based on data features
                    entropy = data_features.get('entropy', 0.5)
                    if entropy > 0.7:
                        # High entropy: use more varied parameters
                        params[param_name] = random.randint(min_val, max_val)
                    else:
                        # Low entropy: use conservative parameters
                        range_mid = (min_val + max_val) // 2
                        range_reduced = (max_val - min_val) // 4
                        params[param_name] = random.randint(
                            max(min_val, range_mid - range_reduced),
                            min(max_val, range_mid + range_reduced)
                        )
                elif param_type == 'float':
                    min_val = param_info.get('min', 0.0)
                    max_val = param_info.get('max', 1.0)
                    params[param_name] = random.uniform(min_val, max_val)
                elif param_type == 'bool':
                    params[param_name] = random.choice([True, False])

        return params

    def _fallback_proposal(self, current_state: Optional[State]) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal when heuristic strategy fails."""
        if self.operations_registry:
            operations = self.operations_registry.list_operations()
            operation_name = random.choice(operations)
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            data_features = self._analyze_data_patterns(current_state) if current_state else {}
            params = self._generate_operation_params(metadata, data_features)
            return (operation_name, params)
        else:
            # Hardcoded fallback
            return ('xor_constant', {'constant': random.randint(1, 255)})

    def accept(self, new_state: State) -> bool:
        """Accept state and update learning data."""
        accepted = new_state.score > self.best_score

        if accepted:
            self.best_score = new_state.score
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        # Update operation performance learning
        if self.operation_history and self.enable_adaptive_learning:
            last_operation, last_params, previous_score = self.operation_history[-1]

            # Calculate performance improvement
            score_change = new_state.score - previous_score
            normalized_change = max(-1.0, min(1.0, score_change / 10.0))  # Normalize to [-1, 1]

            # Update operation performance with learning rate
            current_performances = self.operation_performance.get(last_operation, [])
            current_performances.append(normalized_change)

            # Keep only recent performances
            if len(current_performances) > 100:
                current_performances = current_performances[-50:]

            self.operation_performance[last_operation] = current_performances

            # Update pattern learning
            if self.enable_pattern_recognition and hasattr(self, '_last_data_features'):
                self._update_pattern_learning(last_operation, normalized_change, self._last_data_features)

        return accepted

    def _update_pattern_learning(self, operation_name: str, performance: float,
                               data_features: Dict[str, float]) -> None:
        """Update pattern-based learning."""
        pattern_key = self._create_pattern_key(data_features)

        if pattern_key not in self.pattern_memory:
            self.pattern_memory[pattern_key] = {
                'operations': defaultdict(list),
                'confidence': 0.0,
                'occurrences': 0
            }

        pattern_data = self.pattern_memory[pattern_key]
        pattern_data['operations'][operation_name].append(performance)
        pattern_data['occurrences'] += 1

        # Update operation effectiveness for this pattern
        op_performances = pattern_data['operations'][operation_name]
        if len(op_performances) >= 5:
            # Use average of recent performances
            recent_performances = op_performances[-10:]
            avg_performance = sum(recent_performances) / len(recent_performances)
            pattern_data[operation_name] = avg_performance

        # Update confidence based on number of occurrences
        pattern_data['confidence'] = min(1.0, pattern_data['occurrences'] / 20.0)

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.operation_history.clear()
        self.pattern_memory.clear()