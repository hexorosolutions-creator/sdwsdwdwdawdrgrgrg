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
        self.load_config()
        self.population = None
        self.operations_registry = None
        self.current_state = None

    def load_config(self) -> None:
        """Load configuration from YAML file with fallbacks."""
        config_path = Path("config/strategies/strategy_genetic.yaml")

        try:
            with open(config_path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            # Load parameters from YAML with defaults
            params = yaml_config.get('parameters', {})
            self.population_size = params.get('population_size', 50)
            self.crossover_rate = params.get('crossover_rate', 0.8)
            self.mutation_rate = params.get('mutation_rate', 0.1)
            self.elite_size = params.get('elite_size', 5)
            self.max_sequence_length = params.get('max_sequence_length', 10)

            # Load selection parameters
            selection = yaml_config.get('selection', {})
            self.selection_method = selection.get('method', 'tournament')
            self.tournament_size = selection.get('tournament_size', 3)

            # Load crossover parameters
            crossover = yaml_config.get('crossover', {})
            self.crossover_type = crossover.get('type', 'single_point')

            # Load mutation parameters
            mutation = yaml_config.get('mutation', {})
            self.mutation_type = mutation.get('type', 'point')
            self.mutation_strength = mutation.get('strength', 0.2)

            # Merge with existing config dict
            self.config.update(yaml_config)

        except FileNotFoundError:
            # Create default config file
            self._create_default_config()
            # Use default values
            self.population_size = 50
            self.crossover_rate = 0.8
            self.mutation_rate = 0.1
            self.elite_size = 5
            self.max_sequence_length = 10
            self.selection_method = 'tournament'
            self.tournament_size = 3
            self.crossover_type = 'single_point'
            self.mutation_type = 'point'
            self.mutation_strength = 0.2
        except yaml.YAMLError as e:
            # Log error and use defaults
            print(f"Warning: Error loading genetic config: {e}")
            self.population_size = 50
            self.crossover_rate = 0.8
            self.mutation_rate = 0.1
            self.elite_size = 5
            self.max_sequence_length = 10

    def _create_default_config(self) -> None:
        """Create default genetic strategy config file."""
        config_dir = Path("config/strategies")
        config_dir.mkdir(parents=True, exist_ok=True)

        default_config = {
            'name': 'Genetic Algorithm',
            'description': 'Evolutionary search with crossover and mutation',
            'parameters': {
                'population_size': 50,
                'crossover_rate': 0.8,
                'mutation_rate': 0.1,
                'elite_size': 5,
                'max_sequence_length': 10,
                'max_generations': 1000
            },
            'selection': {
                'method': 'tournament',  # Options: 'tournament', 'roulette_wheel', 'rank_based'
                'tournament_size': 3,
                'pressure': 2.0  # Selection pressure
            },
            'crossover': {
                'type': 'single_point',  # Options: 'single_point', 'two_point', 'uniform'
                'probability': 0.8
            },
            'mutation': {
                'type': 'point',  # Options: 'point', 'insert', 'delete', 'swap'
                'rate': 0.1,
                'strength': 0.2
            },
            'diversity': {
                'enabled': True,
                'threshold': 0.1,
                'penalty': 0.5
            }
        }

        config_path = config_dir / "strategy_genetic.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)

    def set_operations_registry(self, operations_registry) -> None:
        """Set the operations registry for accessing available operations."""
        self.operations_registry = operations_registry

    def propose(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Propose operation using genetic algorithm."""
        if self.operations_registry is None:
            return self._fallback_proposal(current_state)

        try:
            # Initialize population if needed
            if self.population is None:
                self.population = Population(self.population_size, self.operations_registry)
                self.current_state = current_state
                self.population.initialize(current_state, self.max_sequence_length)

            # Check if we should evolve the population
            if self.iteration_count % 10 == 0:  # Evolve every 10 iterations
                self.evolve_population()

            # Get best individual and propose its next operation
            best_individual = self.population.get_best_individual()
            if best_individual and best_individual.operations:
                # Return the next operation from the best individual
                if len(best_individual.operations) > 0:
                    next_op_index = min(self.iteration_count % len(best_individual.operations),
                                       len(best_individual.operations) - 1)
                    return best_individual.operations[next_op_index]

            # Fallback to random operation
            return self._fallback_proposal(current_state)

        except Exception as e:
            print(f"Warning: Genetic strategy failed, using fallback: {e}")
            return self._fallback_proposal(current_state)

    def evolve_population(self) -> None:
        """Evolve the population by one generation."""
        if not self.population or len(self.population.individuals) == 0:
            return

        # Selection - choose parents
        parents = self.select_parents()

        # Create new population through crossover and mutation
        new_individuals = []

        # Elitism - keep best individuals
        elite = self.get_elite_individuals()
        new_individuals.extend(elite)

        # Generate offspring
        while len(new_individuals) < self.population_size:
            if random.random() < self.crossover_rate and len(parents) >= 2:
                # Crossover
                parent1, parent2 = random.sample(parents, 2)
                offspring = self.crossover(parent1, parent2)
            else:
                # Clone parent
                parent = random.choice(parents)
                offspring = parent.copy()

            # Mutation
            if random.random() < self.mutation_rate:
                self.mutate(offspring)

            new_individuals.append(offspring)

        # Update population
        self.population.individuals = new_individuals[:self.population_size]
        self.population.generation += 1

        # Re-evaluate fitness
        self.population.evaluate_fitness(self.current_state)

        # Age individuals
        for individual in self.population.individuals:
            individual.age += 1

    def select_parents(self) -> List[Individual]:
        """Select parents for reproduction."""
        if self.selection_method == 'tournament':
            return self.tournament_selection()
        elif self.selection_method == 'roulette_wheel':
            return self.roulette_wheel_selection()
        elif self.selection_method == 'rank_based':
            return self.rank_based_selection()
        else:
            return self.tournament_selection()

    def tournament_selection(self) -> List[Individual]:
        """Tournament selection."""
        parents = []
        valid_individuals = [ind for ind in self.population.individuals if ind.state is not None]

        for _ in range(self.population_size):
            tournament = random.sample(valid_individuals,
                                      min(self.tournament_size, len(valid_individuals)))
            winner = max(tournament, key=lambda ind: ind.fitness)
            parents.append(winner)

        return parents

    def roulette_wheel_selection(self) -> List[Individual]:
        """Roulette wheel selection."""
        parents = []
        valid_individuals = [ind for ind in self.population.individuals if ind.state is not None]

        # Calculate total fitness (shift to avoid negative values)
        min_fitness = min((ind.fitness for ind in valid_individuals), default=0)
        adjusted_fitnesses = [ind.fitness - min_fitness + 1 for ind in valid_individuals]
        total_fitness = sum(adjusted_fitnesses)

        if total_fitness == 0:
            return random.choices(valid_individuals, k=self.population_size)

        # Calculate probabilities
        probabilities = [fitness / total_fitness for fitness in adjusted_fitnesses]

        for _ in range(self.population_size):
            parent = random.choices(valid_individuals, weights=probabilities)[0]
            parents.append(parent)

        return parents

    def rank_based_selection(self) -> List[Individual]:
        """Rank-based selection."""
        parents = []
        valid_individuals = [ind for ind in self.population.individuals if ind.state is not None]

        # Sort by fitness
        sorted_individuals = sorted(valid_individuals, key=lambda ind: ind.fitness)
        n = len(sorted_individuals)

        # Calculate rank-based probabilities
        ranks = list(range(1, n + 1))
        total_rank = sum(ranks)
        probabilities = [rank / total_rank for rank in ranks]

        for _ in range(self.population_size):
            parent = random.choices(sorted_individuals, weights=probabilities)[0]
            parents.append(parent)

        return parents

    def get_elite_individuals(self) -> List[Individual]:
        """Get elite individuals to preserve."""
        valid_individuals = [ind for ind in self.population.individuals if ind.state is not None]
        sorted_individuals = sorted(valid_individuals, key=lambda ind: ind.fitness, reverse=True)
        return [ind.copy() for ind in sorted_individuals[:self.elite_size]]

    def crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Perform crossover between two parents."""
        if len(parent1.operations) == 0 and len(parent2.operations) == 0:
            return Individual()

        if self.crossover_type == 'single_point':
            return self.single_point_crossover(parent1, parent2)
        elif self.crossover_type == 'two_point':
            return self.two_point_crossover(parent1, parent2)
        elif self.crossover_type == 'uniform':
            return self.uniform_crossover(parent1, parent2)
        else:
            return self.single_point_crossover(parent1, parent2)

    def single_point_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Single-point crossover."""
        if len(parent1.operations) == 0:
            return parent2.copy()
        if len(parent2.operations) == 0:
            return parent1.copy()

        # Choose crossover point
        max_point = min(len(parent1.operations), len(parent2.operations))
        if max_point == 0:
            return Individual()

        crossover_point = random.randint(1, max_point)

        # Create offspring
        offspring_operations = (parent1.operations[:crossover_point] +
                              parent2.operations[crossover_point:])
        return Individual(offspring_operations)

    def two_point_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Two-point crossover."""
        if len(parent1.operations) == 0:
            return parent2.copy()
        if len(parent2.operations) == 0:
            return parent1.copy()

        max_point = min(len(parent1.operations), len(parent2.operations))
        if max_point < 2:
            return self.single_point_crossover(parent1, parent2)

        # Choose two crossover points
        point1 = random.randint(1, max_point - 1)
        point2 = random.randint(point1 + 1, max_point)

        # Create offspring
        offspring_operations = (parent1.operations[:point1] +
                              parent2.operations[point1:point2] +
                              parent1.operations[point2:])
        return Individual(offspring_operations)

    def uniform_crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """Uniform crossover."""
        max_length = max(len(parent1.operations), len(parent2.operations))
        offspring_operations = []

        for i in range(max_length):
            if i < len(parent1.operations) and i < len(parent2.operations):
                # Choose from either parent
                if random.random() < 0.5:
                    offspring_operations.append(parent1.operations[i])
                else:
                    offspring_operations.append(parent2.operations[i])
            elif i < len(parent1.operations):
                offspring_operations.append(parent1.operations[i])
            else:
                offspring_operations.append(parent2.operations[i])

        return Individual(offspring_operations)

    def mutate(self, individual: Individual) -> None:
        """Mutate an individual."""
        if len(individual.operations) == 0:
            # Add random operation
            available_operations = self.operations_registry.list_operations()
            operation_name = random.choice(available_operations)
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = self._generate_operation_params(metadata)
            individual.append((operation_name, params))
            return

        if self.mutation_type == 'point':
            self.point_mutation(individual)
        elif self.mutation_type == 'insert':
            self.insert_mutation(individual)
        elif self.mutation_type == 'delete':
            self.delete_mutation(individual)
        elif self.mutation_type == 'swap':
            self.swap_mutation(individual)
        else:
            self.point_mutation(individual)

    def point_mutation(self, individual: Individual) -> None:
        """Point mutation - replace random operation."""
        if individual.operations:
            mutate_index = random.randint(0, len(individual.operations) - 1)
            available_operations = self.operations_registry.list_operations()
            operation_name = random.choice(available_operations)
            metadata = self.operations_registry.get_operation_metadata(operation_name)
            params = self._generate_operation_params(metadata)
            individual.operations[mutate_index] = (operation_name, params)

    def insert_mutation(self, individual: Individual) -> None:
        """Insert mutation - add new operation."""
        available_operations = self.operations_registry.list_operations()
        operation_name = random.choice(available_operations)
        metadata = self.operations_registry.get_operation_metadata(operation_name)
        params = self._generate_operation_params(metadata)

        insert_index = random.randint(0, len(individual.operations))
        individual.operations.insert(insert_index, (operation_name, params))

    def delete_mutation(self, individual: Individual) -> None:
        """Delete mutation - remove random operation."""
        if len(individual.operations) > 1:
            delete_index = random.randint(0, len(individual.operations) - 1)
            del individual.operations[delete_index]

    def swap_mutation(self, individual: Individual) -> None:
        """Swap mutation - swap adjacent operations."""
        if len(individual.operations) > 1:
            swap_index = random.randint(0, len(individual.operations) - 2)
            individual.operations[swap_index], individual.operations[swap_index + 1] = \
                individual.operations[swap_index + 1], individual.operations[swap_index]

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

    def _fallback_proposal(self, current_state: State) -> Tuple[str, Dict[str, Any]]:
        """Fallback proposal when genetic algorithm fails."""
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
        """Accept based on fitness improvement."""
        # Update current state for population evaluation
        self.current_state = new_state

        # Re-evaluate population with new state occasionally
        if self.population and self.iteration_count % 50 == 0:
            self.population.evaluate_fitness(new_state)

        return new_state.score > self.best_score

    def reset(self) -> None:
        """Reset the strategy state."""
        super().reset()
        self.population = None
        self.current_state = None