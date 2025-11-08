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
        self.load_config()
        self.root_node = None
        self.operations_registry = None

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_path = Path("config/strategies/strategy_mcts.yaml")

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Load parameters from YAML with defaults
            params = yaml_config.get('parameters', {})
            self.exploration_constant = params.get('exploration_constant', 1.4)
            self.simulation_count = params.get('simulation_count', 100)
            self.rollout_depth = params.get('rollout_depth', 10)
            self.max_tree_depth = params.get('max_tree_depth', 50)

            # Load tree management parameters
            tree_mgmt = yaml_config.get('tree_management', {})
            self.expansion_policy = tree_mgmt.get('expansion_policy', 'uct')
            self.max_children = tree_mgmt.get('max_children', 20)
            self.prune_tree = tree_mgmt.get('prune_tree', True)
            self.prune_threshold = tree_mgmt.get('prune_threshold', 0.01)

            # Load simulation parameters
            sim = yaml_config.get('simulation', {})
            self.simulation_policy = sim.get('policy', 'random')
            self.early_termination = sim.get('early_termination', True)
            self.early_termination_threshold = sim.get('early_termination_threshold', 0.8)

            # Load backpropagation parameters
            backprop = yaml_config.get('backpropagation', {})
            self.value_function = backprop.get('value_function', 'score')
            self.discount_factor = backprop.get('discount_factor', 0.9)
            self.backpropagation_mode = backprop.get('backpropagation_mode', 'average')

            # Load selection parameters
            selection = yaml_config.get('selection', {})
            self.final_selection = selection.get('final_selection', 'most_visited')

            # Load memory management parameters
            memory = yaml_config.get('memory', {})
            self.max_tree_nodes = memory.get('max_tree_nodes', 10000)
            self.cleanup_policy = memory.get('cleanup_policy', 'least_visited')
            self.cleanup_interval = memory.get('cleanup_interval', 100)

            # Load parallel parameters
            parallel = yaml_config.get('parallel', {})
            self.parallel_simulations = parallel.get('parallel_simulations', False)
            self.num_workers = parallel.get('num_workers', 4)

            # Merge with existing config dict
            self.config.update(yaml_config)

        except FileNotFoundError:
            # Use default values if config file missing
            self.exploration_constant = 1.4
            self.simulation_count = 100
            self.rollout_depth = 10
            self.max_tree_depth = 50
            self.expansion_policy = 'uct'
            self.max_children = 20
            self.prune_tree = True
            self.prune_threshold = 0.01
            self.simulation_policy = 'random'
            self.early_termination = True
            self.early_termination_threshold = 0.8
            self.value_function = 'score'
            self.discount_factor = 0.9
            self.backpropagation_mode = 'average'
            self.final_selection = 'most_visited'
            self.max_tree_nodes = 10000
            self.cleanup_policy = 'least_visited'
            self.cleanup_interval = 100
            self.parallel_simulations = False
            self.num_workers = 4
        except yaml.YAMLError as e:
            # Log error and use defaults
            print(f"Warning: Error loading MCTS config: {e}")
            self.exploration_constant = 1.4
            self.simulation_count = 100
            self.rollout_depth = 10
            self.max_tree_depth = 50

    def set_operations_registry(self, operations_registry) -> None:
        """Set the operations registry for accessing available operations."""
        self.operations_registry = operations_registry

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using MCTS."""
        if self.operations_registry is None:
            # Fallback to simple random choice if no registry
            return self._fallback_proposal(current_state)

        try:
            # Initialize root node if needed
            if self.root_node is None or self.iteration_count % self.cleanup_interval == 0:
                self.root_node = TreeNode(current_state)
                self._initialize_root_node()

            # Run MCTS simulations
            for _ in range(self.simulation_count):
                if self._get_tree_size() > self.max_tree_nodes:
                    self._prune_tree()

                # Selection phase
                selected_node = self.select_node()

                # Expansion phase
                if not selected_node.is_terminal():
                    expanded_node = self.expand_node(selected_node)
                    if expanded_node:
                        # Simulation phase
                        result = self.simulate(expanded_node)

                        # Backpropagation phase
                        self.backpropagate(expanded_node, result)

            # Get best move based on configuration
            return self.get_best_move()

        except Exception as e:
            # Fallback to simpler strategy if MCTS fails
            print(f"Warning: MCTS failed, using fallback: {e}")
            return self._fallback_proposal(current_state)

    def _initialize_root_node(self) -> None:
        """Initialize root node with available operations."""
        if self.operations_registry:
            operations = self.operations_registry.list_operations()
            self.root_node.untried_operations = operations[:self.max_children]

    def select_node(self) -> TreeNode:
        """Select a node for expansion using UCT."""
        node = self.root_node

        while not node.is_terminal() and node.is_fully_expanded():
            node = node.get_best_child(self.exploration_constant)

        return node

    def expand_node(self, node: TreeNode) -> Optional[TreeNode]:
        """Expand a node by trying an untried operation."""
        if not node.untried_operations:
            return None

        # Select an untried operation
        operation_name = node.untried_operations.pop(0)

        try:
            # Get operation function and parameters
            operation_func = self.operations_registry.get_operation(operation_name)
            metadata = self.operations_registry.get_operation_metadata(operation_name)

            # Generate valid parameters for the operation
            params = self._generate_operation_params(metadata)

            # Apply operation to create new state
            new_data, inverse_func, op_metadata = operation_func(node.state.data, **params)
            new_state = State(new_data, self.operations_registry)

            # Create child node
            child_node = TreeNode(new_state, parent=node, operation=(operation_name, params))
            node.children[operation_name] = child_node

            return child_node

        except Exception as e:
            # Skip this operation if it fails
            print(f"Warning: Operation {operation_name} failed during expansion: {e}")
            return None

    def simulate(self, node: TreeNode) -> float:
        """Run a simulation from the given node."""
        current_state = copy.deepcopy(node.state)
        depth = 0
        total_score = 0.0

        while not current_state.is_terminal() and depth < self.rollout_depth:
            try:
                # Select operation based on simulation policy
                if self.simulation_policy == 'random':
                    operation_name = random.choice(self.operations_registry.list_operations())
                elif self.simulation_policy == 'greedy':
                    operation_name = self._select_greedy_operation(current_state)
                else:  # guided
                    operation_name = self._select_guided_operation(current_state)

                # Apply operation
                operation_func = self.operations_registry.get_operation(operation_name)
                metadata = self.operations_registry.get_operation_metadata(operation_name)
                params = self._generate_operation_params(metadata)

                new_data, _, _ = operation_func(current_state.data, **params)
                current_state = State(new_data, self.operations_registry)

                total_score += current_state.score

                # Early termination check
                if self.early_termination and current_state.score > self.early_termination_threshold:
                    break

                depth += 1

            except Exception:
                # Skip operations that fail
                depth += 1
                continue

        # Return normalized score
        return total_score / max(1, depth)

    def backpropagate(self, node: TreeNode, result: float) -> None:
        """Backpropagate simulation results up the tree."""
        current_node = node

        while current_node is not None:
            # Apply value function transformation
            if self.value_function == 'improvement':
                # Compare to parent state if available
                if current_node.parent:
                    improvement = current_node.state.score - current_node.parent.state.score
                    transformed_result = improvement
                else:
                    transformed_result = result
            elif self.value_function == 'rank':
                # Normalize to [0, 1] range based on expected score range
                transformed_result = max(0, min(1, (result + 1) / 2))
            else:  # score
                transformed_result = result

            # Apply discount factor
            if current_node != node:
                transformed_result *= self.discount_factor

            current_node.update(transformed_result)
            current_node = current_node.parent

    def get_best_move(self) -> Tuple[str, Dict[str, Any]]:
        """Get the best move based on final selection policy."""
        if not self.root_node or not self.root_node.children:
            return self._fallback_proposal(self.root_node.state if self.root_node else None)

        if self.final_selection == 'most_visited':
            best_child = self.root_node.get_most_visited_child()
        elif self.final_selection == 'highest_value':
            best_child = self.root_node.get_highest_value_child()
        elif self.final_selection == 'robust':
            # Choose child with good balance of visits and value
            best_child = None
            best_score = float('-inf')
            for child in self.root_node.children.values():
                if child.visits > 10:  # Minimum visits threshold
                    score = (child.value / child.visits) * math.log(child.visits)
                    if score > best_score:
                        best_score = score
                        best_child = child
            if best_child is None:
                best_child = self.root_node.get_most_visited_child()
        else:
            best_child = self.root_node.get_most_visited_child()

        if best_child.operation:
            return best_child.operation
        else:
            return self._fallback_proposal(best_child.state)

    def _generate_operation_params(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate valid parameters for an operation."""
        params = {}

        # Get parameter definitions from metadata
        param_defs = metadata.get('parameters', {})

        for param_name, param_info in param_defs.items():
            if param_info.get('required', False):
                # Generate reasonable default values based on parameter type
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

    def _select_greedy_operation(self, state: State) -> str:
        """Select operation based on expected improvement (greedy)."""
        operations = self.operations_registry.list_operations()
        best_op = random.choice(operations)
        best_expected_score = float('-inf')

        for op_name in operations[:5]:  # Sample first 5 operations
            try:
                metadata = self.operations_registry.get_operation_metadata(op_name)
                # Simple heuristic based on operation category and metadata
                expected_score = metadata.get('expected_improvement', 0.0)
                if expected_score > best_expected_score:
                    best_expected_score = expected_score
                    best_op = op_name
            except Exception:
                continue

        return best_op

    def _select_guided_operation(self, state: State) -> str:
        """Select operation using guided exploration."""
        operations = self.operations_registry.list_operations()
        # Use temperature to balance exploration/exploitation
        temp = max(0.1, 1.0 - self.iteration_count / 1000.0)

        if random.random() < temp:
            # Explore
            return random.choice(operations)
        else:
            # Exploit
            return self._select_greedy_operation(state)

    def _get_tree_size(self) -> int:
        """Get the current size of the tree."""
        if not self.root_node:
            return 0

        def count_nodes(node):
            count = 1
            for child in node.children.values():
                count += count_nodes(child)
            return count

        return count_nodes(self.root_node)

    def _prune_tree(self) -> None:
        """Prune the tree to stay within memory limits."""
        if not self.root_node or not self.prune_tree:
            return

        # Simple pruning: remove least visited nodes at each level
        def prune_nodes(node):
            if len(node.children) > self.max_children:
                # Sort children by visits and keep only the best ones
                sorted_children = sorted(
                    node.children.items(),
                    key=lambda x: x[1].visits
                )

                # Remove worst children
                for op_name, child in sorted_children[:-self.max_children]:
                    if child.visits < self.prune_threshold * max(1, node.visits):
                        del node.children[op_name]

            # Recursively prune children
            for child in list(node.children.values()):
                prune_nodes(child)

        prune_nodes(self.root_node)

    def _fallback_proposal(self, current_state: Optional[State]) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal when MCTS fails."""
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
        """Accept based on MCTS evaluation."""
        # Reset tree if we're moving to a significantly different state
        if self.root_node and abs(new_state.score - self.root_node.state.score) > 0.5:
            self.root_node = None

        return new_state.score > self.best_score

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.root_node = None