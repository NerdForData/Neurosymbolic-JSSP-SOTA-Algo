"""
Neuro-Symbolic Integration Loop
================================
The central orchestrator of the hybrid system.

Flow:
  1. LLM generates seed sequences (Neuro → generalisation)
  2. Decoder converts sequences to feasible schedules
  3. GA evolves population seeded with LLM output (Symbolic → guarantees)
  4. If GA stagnates, LLM is called again for fresh diversity
  5. Best schedule returned with full audit trail

This implements the "Neuro|Symbolic" pipeline from the NSAI taxonomy:
  Language + Guarantees + Generalisation
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, List

from src.problem.jssp import JSSPInstance, Schedule
from src.llm.llm_client import GroqLLMClient
from src.llm.decoder import decode_population
from src.ga.genetic_algorithm import GeneticAlgorithm, GAConfig, Individual

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class NSResult:
    """
    Complete result from one NS hybrid run.
    Contains the best schedule, all metrics, and the full audit trail.
    """
    instance_name:    str
    best_schedule:    Schedule
    best_fitness:     float
    best_makespan:    float
    optimality_gap:   Optional[float]

    # Timing
    llm_time_sec:     float
    ga_time_sec:      float
    total_time_sec:   float

    # Population info
    n_llm_seeds:      int
    n_llm_calls:      int

    # Convergence
    best_per_gen:     List[float] = field(default_factory=list)
    avg_per_gen:      List[float] = field(default_factory=list)

    # Audit trail
    llm_reasoning:    str = ""

    def summary(self) -> str:
        lines = [
            f"=== NS Hybrid Result: {self.instance_name} ===",
            f"  Makespan      : {self.best_makespan:.0f}",
            f"  Fitness       : {self.best_fitness:.4f}",
        ]
        if self.optimality_gap is not None:
            lines.append(f"  Optimality gap: {self.optimality_gap:.1f}%")
        metrics = self.best_schedule.compute_metrics()
        lines += [
            f"  Tardiness     : {metrics.get('tardiness', 0):.1f}",
            f"  Utilization   : {metrics.get('utilization', 0):.1%}",
            f"  Flow time     : {metrics.get('flowtime', 0):.1f}",
            f"  Energy        : {metrics.get('energy', 0):.1f}",
            f"  LLM seeds     : {self.n_llm_seeds} (calls={self.n_llm_calls})",
            f"  Time          : LLM={self.llm_time_sec:.1f}s, "
            f"GA={self.ga_time_sec:.1f}s, "
            f"total={self.total_time_sec:.1f}s",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# NS Hybrid Solver
# ---------------------------------------------------------------------------

class NSHybridSolver:
    """
    Neuro-Symbolic Hybrid Solver for JSSP.

    Combines:
      - Groq LLM (Llama 3.1 8B) for intelligent population seeding
      - Constraint-aware GA for guaranteed feasible optimisation
    """

    def __init__(
        self,
        llm_client: Optional[GroqLLMClient] = None,
        ga_config:  Optional[GAConfig]      = None,
        n_llm_seeds: int = 5,
        stagnation_threshold: int = 50,   # generations without improvement
        reinject_on_stagnation: bool = True,
    ):
        self.llm    = llm_client or GroqLLMClient()
        self.ga_cfg = ga_config  or GAConfig()
        self.n_llm_seeds            = n_llm_seeds
        self.stagnation_threshold   = stagnation_threshold
        self.reinject_on_stagnation = reinject_on_stagnation

    def solve(self, instance: JSSPInstance, verbose: bool = True) -> NSResult:
        """
        Run the full NS hybrid solve on a JSSP instance.

        Args:
            instance: The JSSP instance to solve
            verbose:  Print progress to stdout

        Returns:
            NSResult with best schedule and full diagnostics
        """
        if verbose:
            print(f"\n{'='*55}")
            print(f"  NS Hybrid Solver: {instance}")
            print(f"{'='*55}")

        total_start = time.time()
        n_llm_calls = 0

        # ── Step 1: LLM seeding ─────────────────────────────────────────
        if verbose:
            print(f"\n[1/3] LLM seeding ({self.n_llm_seeds} sequences)...")

        llm_start = time.time()
        seed_sequences = self.llm.generate_sequences(
            instance, n=self.n_llm_seeds
        )
        n_llm_calls += 1
        llm_time = time.time() - llm_start

        if verbose:
            # Show initial quality of LLM seeds
            seed_schedules = decode_population(seed_sequences, instance)
            seed_fitnesses = [s.fitness() for s in seed_schedules]
            print(f"  LLM seeds decoded. Fitness range: "
                  f"[{min(seed_fitnesses):.4f}, {max(seed_fitnesses):.4f}]")
            print(f"  LLM call time: {llm_time:.2f}s")

        # ── Step 2: GA evolution ────────────────────────────────────────
        if verbose:
            print(f"\n[2/3] GA evolution "
                  f"(pop={self.ga_cfg.population_size}, "
                  f"gens={self.ga_cfg.n_generations})...")

        ga = GeneticAlgorithm(instance, self.ga_cfg)

        # Progress callback for verbose mode
        def _on_progress(gen, best_fit):
            if verbose and gen % 50 == 0:
                print(f"  Gen {gen:4d} | best fitness = {best_fit:.4f}")

        ga_start = time.time()
        best_ind = ga.evolve(
            seed_sequences=seed_sequences,
            progress_callback=_on_progress,
        )
        ga_time = time.time() - ga_start

        # ── Step 3: Stagnation re-injection ────────────────────────────
        if self.reinject_on_stagnation:
            history = ga.best_per_gen
            if len(history) > self.stagnation_threshold:
                recent = history[-self.stagnation_threshold:]
                if max(recent) - min(recent) < 1e-6:
                    if verbose:
                        print(f"\n[3/3] Stagnation detected. "
                              f"Re-injecting LLM seeds...")
                    llm_start2 = time.time()
                    new_seeds = self.llm.generate_sequences(
                        instance, n=self.n_llm_seeds, temperature=0.9
                    )
                    n_llm_calls += 1
                    llm_time += time.time() - llm_start2

                    # Run GA again from combined population
                    ga2 = GeneticAlgorithm(instance, self.ga_cfg)
                    best_ind2 = ga2.evolve(seed_sequences=new_seeds)
                    if best_ind2.fitness < best_ind.fitness:
                        best_ind = best_ind2
                        ga = ga2
                        if verbose:
                            print(f"  Re-injection improved fitness to "
                                  f"{best_ind.fitness:.4f}")
        elif verbose:
            print(f"\n[3/3] No stagnation detected.")

        total_time = time.time() - total_start

        # ── Build result ────────────────────────────────────────────────
        metrics = best_ind.schedule.compute_metrics()
        result = NSResult(
            instance_name  = instance.name,
            best_schedule  = best_ind.schedule,
            best_fitness   = best_ind.fitness,
            best_makespan  = metrics.get("makespan", 0),
            optimality_gap = best_ind.schedule.optimality_gap(),
            llm_time_sec   = llm_time,
            ga_time_sec    = ga_time,
            total_time_sec = total_time,
            n_llm_seeds    = self.n_llm_seeds,
            n_llm_calls    = n_llm_calls,
            best_per_gen   = ga.best_per_gen,
            avg_per_gen    = ga.avg_per_gen,
        )

        if verbose:
            print(f"\n{result.summary()}")

        return result