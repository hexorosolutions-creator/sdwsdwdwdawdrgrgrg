"""
Genetic algorithm strategy for BSEE.
"""

import random
import copy
import yaml
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
from bsee.strategies.base_strategy import BaseStrategy
from bsee.engine.state import State


class Individual:
    """Individual in the genetic algorithm population."""

    def __init__(self, operations: Optional[List[Tuple[str, Dict[str, Any]]]] = None):
        """Initialize an individual with a sequence of operations."""
        self.operations = operations or []
        self.fitness = float('-inf')
        self.age = 0
        self.state = None  # Will be set when operations are applied

    def __len__(self):
        """Get the length of the individual's operation sequence."""
        return len(self.operations)

    def __getitem__(self, index):
        """Get operation at specific index."""
        return self.operations[index]

    def append(self, operation: Tuple[str, Dict[str, Any]]):
        """Add an operation to the individual."""
        self.operations.append(operation)

    def copy(self) -> 'Individual':
        """Create a copy of this individual."""
        new_individual = Individual(copy.deepcopy(self.operations))
        new_individual.fitness = self.fitness
        new_individual.age = self.age
        return new_individual


class Population:
    """Population of individuals in the genetic algorithm."""

    def __init__(self, size: int, operations_registry):
        """Initialize population."""
        self.size = size
        self.individuals: List[Individual] = []
        self.operations_registry = operations_registry
        self.generation = 0
        self.best_individual = None
        self.average_fitness = 0.0

    def initialize(self, initial_state: State, max_sequence_length: int = 10):
        """Initialize population with random individuals."""
        self.individuals = []
        available_operations = self.operations_registry.list_operations()

        for _ in range(self.size):
            individual = Individual()
            sequence_length = random.randint(1, max_sequence_length)

            for _ in range(sequence_length):
                operation_name = random.choice(available_operations)
                metadata = self.operations_registry.get_operation_metadata(operation_name)
                params = self._generate_operation_params(metadata)
                individual.append((operation_name, params))

            self.individuals.append(individual)

        # Evaluate initial fitness
        self.evaluate_fitness(initial_state)

    def evaluate_fitness(self, initial_state: State):
        """Evaluate fitness of all individuals."""
        total_fitness = 0.0
        evaluated_count = 0

        for individual in self.individuals:
            try:
                # Apply operations to get final state
                current_state = copy.deepcopy(initial_state)

                for operation_name, params in individual.operations:
                    operation_func = self.operations_registry.get_operation(operation_name)
                    new_data, _, _ = operation_func(current_state.data, **params)
                    current_state = State(new_data, self.operations_registry)

                individual.state = current_state
                individual.fitness = current_state.score
                total_fitness += individual.fitness
                evaluated_count += 1

            except Exception as e:
                # Penalize individuals that fail during evaluation
                individual.fitness = -1000.0
                individual.state = None
                total_fitness += individual.fitness
                evaluated_count += 1

        # Update population statistics
        self.average_fitness = total_fitness / evaluated_count if evaluated_count > 0 else 0.0

        # Track best individual
        valid_individuals = [ind for ind in self.individuals if ind.state is not None]
        if valid_individuals:
            self.best_individual = max(valid_individuals, key=lambda ind: ind.fitness)

    def get_best_individual(self) -> Optional[Individual]:
        """Get the best individual in the population."""
        if self.best_individual:
            return self.best_individual
        else:
            valid_individuals = [ind for ind in self.individuals if ind.state is not None]
            return max(valid_individuals, key=lambda ind: ind.fitness) if valid_individuals else None

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


class GeneticStrategy(BaseStrategy):
    """Genetic algorithm strategy."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize genetic strategy."""
        super().__init__(config)
        self.population_size = config.get('population_size', 20)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        self.crossover_rate = config.get('crossover_rate', 0.7)

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using genetic operators."""
        operations = [
            ('xor_constant', {'constant': random.randint(1, 255)}),
            ('rotate_left', {'shift': random.randint(1, 7)}),
            ('move_to_front', {}),
            ('shuffle_bytes', {'seed': random.randint(0, 10000)})
        ]
        return random.choice(operations)

    def accept(self, new_state: State) -> bool:
        """Accept based on fitness."""
        return new_state.score > self.best_score