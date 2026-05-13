"""
Genetic Algorithm Engine
========================
Constraint-aware GA for the Job Shop Scheduling Problem.

This is the "Symbolic" half of the Neuro-Symbolic system.

Key design decisions:
  - Chromosome = priority sequence (operation-based encoding)
  - Initial population = LLM seeds + heuristics (no random init)
  - Crossover = Order Crossover (OX) adapted for multiset sequences
  - Mutation  = swap mutation (preserves sequence validity)
  - Selection = tournament selection
  - Constraint guarantee: every individual is valid by encoding design
  - Fitness   = multi-objective weighted score from Schedule.fitness()
"""

import random
import logging
import copy
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

from src.problem.jssp import JSSPInstance, Schedule
from src.llm.decoder import decode_sequence

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GA configuration
# ---------------------------------------------------------------------------

@dataclass
class GAConfig:
    """All hyperparameters for the GA in one place."""
    population_size:  int   = 50
    n_generations:    int   = 200
    crossover_rate:   float = 0.85
    mutation_rate:    float = 0.15
    tournament_size:  int   = 3
    elitism_count:    int   = 2      # top-k individuals always survive
    diversity_inject: int   = 10     # inject new random individuals every N gens
                                     # (prevents premature convergence)
    seed:             Optional[int] = 42


# ---------------------------------------------------------------------------
# Individual
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """One chromosome in the GA population."""
    sequence: List[int]
    fitness:  float = float("inf")
    schedule: Optional[Schedule] = field(default=None, repr=False)

    def evaluate(self, instance: JSSPInstance) -> float:
        """Decode sequence → Schedule → compute fitness."""
        self.schedule = decode_sequence(self.sequence, instance)
        self.fitness  = self.schedule.fitness()
        return self.fitness

    def copy(self) -> "Individual":
        ind = Individual(sequence=self.sequence[:], fitness=self.fitness)
        ind.schedule = self.schedule  # share reference (schedule is read-only)
        return ind


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _ox_crossover(
    parent_a: List[int],
    parent_b: List[int],
    n_jobs: int,
    n_machines: int,
) -> Tuple[List[int], List[int]]:
    """
    Order Crossover (OX) adapted for operation-based encoding.

    Standard OX preserves relative order of elements from one parent
    while filling remaining slots from the other parent.
    Adapted here to handle multisets (each job_id appears n_machines times).
    """
    size = len(parent_a)
    # Pick two random cut points
    c1, c2 = sorted(random.sample(range(size), 2))

    def _build_child(p1: List[int], p2: List[int]) -> List[int]:
        # Copy the segment from p1
        child = [None] * size
        child[c1:c2] = p1[c1:c2]

        # Count how many of each job_id already placed
        from collections import Counter
        placed = Counter(child[c1:c2])
        remaining_needed = {j: n_machines - placed[j] for j in range(n_jobs)}

        # Fill remaining positions with p2 elements in order
        fill_values = [x for x in p2 if remaining_needed[x] > 0
                       and (remaining_needed.__setitem__(x, remaining_needed[x]-1) or True)]
        # above trick mutates remaining_needed inline; redo cleanly:
        remaining_needed = {j: n_machines - placed[j] for j in range(n_jobs)}
        fill_values = []
        for x in p2:
            if remaining_needed.get(x, 0) > 0:
                fill_values.append(x)
                remaining_needed[x] -= 1

        # Place fill values into child positions outside [c1, c2]
        fill_idx = 0
        for i in list(range(c2, size)) + list(range(0, c1)):
            if fill_idx < len(fill_values):
                child[i] = fill_values[fill_idx]
                fill_idx += 1

        return child

    child_a = _build_child(parent_a, parent_b)
    child_b = _build_child(parent_b, parent_a)
    return child_a, child_b


def _swap_mutation(sequence: List[int]) -> List[int]:
    """
    Swap two random positions in the sequence.
    Preserves the multiset property (each job still appears n_machines times).
    """
    seq = sequence[:]
    i, j = random.sample(range(len(seq)), 2)
    seq[i], seq[j] = seq[j], seq[i]
    return seq


def _insert_mutation(sequence: List[int]) -> List[int]:
    """
    Remove an element and insert it at a random position.
    More disruptive than swap -- used for diversity.
    """
    seq = sequence[:]
    i = random.randrange(len(seq))
    j = random.randrange(len(seq))
    elem = seq.pop(i)
    seq.insert(j, elem)
    return seq


def _tournament_select(
    population: List[Individual],
    tournament_size: int,
) -> Individual:
    """Select the best individual from a random tournament."""
    contestants = random.sample(population, min(tournament_size, len(population)))
    return min(contestants, key=lambda ind: ind.fitness)


# ---------------------------------------------------------------------------
# GA Engine
# ---------------------------------------------------------------------------

class GeneticAlgorithm:
    """
    Constraint-aware Genetic Algorithm for JSSP.

    The GA never generates infeasible solutions because:
      1. The operation-based encoding is valid by construction
      2. OX crossover and swap mutation preserve the encoding invariant
      3. The decoder (Giffler-Thompson) always produces feasible schedules

    This is the symbolic guarantee layer of the NS system.
    """

    def __init__(self, instance: JSSPInstance, config: GAConfig = None):
        self.instance = instance
        self.config   = config or GAConfig()
        self.n_jobs   = instance.num_jobs
        self.n_mach   = instance.num_machines
        self.seq_len  = self.n_jobs * self.n_mach

        # History tracking for benchmarking
        self.best_per_gen:    List[float] = []
        self.avg_per_gen:     List[float] = []
        self.best_individual: Optional[Individual] = None

        if self.config.seed is not None:
            random.seed(self.config.seed)

    # ── Initialisation ──────────────────────────────────────────────────

    def _random_sequence(self) -> List[int]:
        seq = list(range(self.n_jobs)) * self.n_mach
        random.shuffle(seq)
        return seq

    def initialise_population(
        self,
        seed_sequences: Optional[List[List[int]]] = None,
    ) -> List[Individual]:
        """
        Build initial population from LLM seeds + random fill.

        Args:
            seed_sequences: Sequences from the LLM module (can be None)

        Returns:
            Evaluated population sorted by fitness
        """
        population = []

        # Add LLM seeds first
        if seed_sequences:
            for seq in seed_sequences:
                ind = Individual(sequence=seq[:])
                ind.evaluate(self.instance)
                population.append(ind)
            logger.info(f"  {len(seed_sequences)} LLM seed(s) added to population")

        # Fill remaining slots with random sequences
        remaining = self.config.population_size - len(population)
        for _ in range(remaining):
            ind = Individual(sequence=self._random_sequence())
            ind.evaluate(self.instance)
            population.append(ind)

        population.sort(key=lambda x: x.fitness)
        return population

    # ── Main evolution loop ──────────────────────────────────────────────

    def evolve(
        self,
        seed_sequences: Optional[List[List[int]]] = None,
        progress_callback: Optional[Callable] = None,
    ) -> Individual:
        """
        Run the full GA evolution.

        Args:
            seed_sequences:    LLM-generated sequences for population seeding
            progress_callback: Optional fn(gen, best_fitness) for live updates

        Returns:
            The best Individual found across all generations
        """
        cfg = self.config
        logger.info(f"GA starting: pop={cfg.population_size}, "
                    f"gens={cfg.n_generations}, "
                    f"cx={cfg.crossover_rate}, mut={cfg.mutation_rate}")

        # Initialise
        population = self.initialise_population(seed_sequences)
        self.best_individual = population[0].copy()

        for gen in range(cfg.n_generations):
            new_population = []

            # Elitism: carry top individuals unchanged
            elite = [ind.copy() for ind in population[:cfg.elitism_count]]
            new_population.extend(elite)

            # Fill rest via crossover + mutation
            while len(new_population) < cfg.population_size:
                parent_a = _tournament_select(population, cfg.tournament_size)
                parent_b = _tournament_select(population, cfg.tournament_size)

                # Crossover
                if random.random() < cfg.crossover_rate:
                    seq_a, seq_b = _ox_crossover(
                        parent_a.sequence, parent_b.sequence,
                        self.n_jobs, self.n_mach
                    )
                else:
                    seq_a = parent_a.sequence[:]
                    seq_b = parent_b.sequence[:]

                # Mutation
                for seq in [seq_a, seq_b]:
                    if random.random() < cfg.mutation_rate:
                        seq[:] = _swap_mutation(seq)
                    if random.random() < cfg.mutation_rate * 0.5:
                        seq[:] = _insert_mutation(seq)

                # Evaluate and add children
                for seq in [seq_a, seq_b]:
                    if len(new_population) < cfg.population_size:
                        ind = Individual(sequence=seq)
                        ind.evaluate(self.instance)
                        new_population.append(ind)

            # Diversity injection: replace worst individuals with fresh randoms
            if gen % cfg.diversity_inject == 0 and gen > 0:
                for i in range(cfg.diversity_inject):
                    idx = cfg.population_size - 1 - i
                    new_population[idx] = Individual(
                        sequence=self._random_sequence()
                    )
                    new_population[idx].evaluate(self.instance)

            # Sort and track stats
            new_population.sort(key=lambda x: x.fitness)
            population = new_population

            best_fitness = population[0].fitness
            avg_fitness  = sum(i.fitness for i in population) / len(population)
            self.best_per_gen.append(best_fitness)
            self.avg_per_gen.append(avg_fitness)

            # Update global best
            if best_fitness < self.best_individual.fitness:
                self.best_individual = population[0].copy()

            # Progress callback
            if progress_callback:
                progress_callback(gen, best_fitness)

            # Logging every 25 generations
            if gen % 25 == 0 or gen == cfg.n_generations - 1:
                gap = None
                if self.best_individual.schedule is not None:
                    gap = self.best_individual.schedule.optimality_gap()
                gap_str = f", gap={gap:.1f}%" if gap is not None else ""
                logger.info(
                    f"  Gen {gen:4d}: best={best_fitness:.4f}, "
                    f"avg={avg_fitness:.4f}{gap_str}"
                )

        logger.info(
            f"GA complete. Best fitness={self.best_individual.fitness:.4f}"
        )
        return self.best_individual

    def convergence_stats(self) -> dict:
        """Return convergence history for plotting."""
        return {
            "best_per_gen": self.best_per_gen,
            "avg_per_gen":  self.avg_per_gen,
            "n_generations": len(self.best_per_gen),
        }