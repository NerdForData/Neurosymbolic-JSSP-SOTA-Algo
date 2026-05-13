"""
Benchmark Runner
================
Unified harness that runs every solver on every instance,
records all metrics, and saves results to CSV.

Fully reproducible: same config + same seed = identical results.

Usage:
    from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig
    runner = BenchmarkRunner(config)
    results = runner.run()
    runner.save("results/benchmark.csv")
"""

import os
import csv
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional

import pandas as pd

from src.problem.loader import load_instance
from src.problem.jssp import JSSPInstance
from src.benchmark.baseline import (
    DispatchingRuleSolver, PureGASolver, PSOSolver,
    ORToolsSolver, SolverResult
)
from src.ga.genetic_algorithm import GAConfig
from src.ns_solver import NSHybridSolver
from src.llm.llm_client import GroqLLMClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """
    Full benchmark configuration.
    Controls which instances and solvers to run, and with what parameters.
    """

    # Instances to benchmark (start small, expand for full report)
    instances: List[str] = field(default_factory=lambda: [
        "ft06", "ft10",               # Tier 1: development
        "ta01", "ta02", "ta03",       # Tier 2: medium (add more after download)
    ])

    # Number of independent runs per solver per instance
    # (for statistical significance -- 5 minimum for Wilcoxon test)
    n_runs: int = 5

    # GA settings (shared by PureGA and NS Hybrid)
    ga_population:  int = 50
    ga_generations: int = 200

    # PSO settings
    pso_particles:  int = 30
    pso_iterations: int = 200

    # OR-Tools time limit per instance
    ortools_time_limit: int = 60

    # Which solvers to run
    run_dispatching: bool = True
    run_pure_ga:     bool = True
    run_pso:         bool = True
    run_ortools:     bool = True
    run_ns_hybrid:   bool = True

    # Number of LLM seeds for NS Hybrid
    n_llm_seeds: int = 5

    # Output directory
    output_dir: str = "results"

    # Random seed base (each run uses seed + run_index for reproducibility)
    base_seed: int = 42


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """
    Runs all configured solvers on all instances and collects results.
    """

    def __init__(self, config: BenchmarkConfig = None):
        self.config  = config or BenchmarkConfig()
        self.results: List[SolverResult] = []

    def _build_solvers(self, run_idx: int) -> List:
        """Build fresh solver instances for each run (with unique seeds)."""
        cfg = self.config
        seed = cfg.base_seed + run_idx

        ga_cfg = GAConfig(
            population_size = cfg.ga_population,
            n_generations   = cfg.ga_generations,
            seed            = seed,
        )

        solvers = []

        if cfg.run_dispatching:
            for rule in ["SPT", "LPT", "EDD", "MWR"]:
                solvers.append(DispatchingRuleSolver(rule=rule))

        if cfg.run_pure_ga:
            solvers.append(PureGASolver(config=ga_cfg))

        if cfg.run_pso:
            solvers.append(PSOSolver(
                n_particles  = cfg.pso_particles,
                n_iterations = cfg.pso_iterations,
                seed         = seed,
            ))

        if cfg.run_ortools:
            solvers.append(ORToolsSolver(
                time_limit_sec = cfg.ortools_time_limit
            ))

        if cfg.run_ns_hybrid:
            solvers.append(NSHybridSolver(
                llm_client  = GroqLLMClient(),
                ga_config   = ga_cfg,
                n_llm_seeds = cfg.n_llm_seeds,
            ))

        return solvers

    def run(self, verbose: bool = True) -> List[SolverResult]:
        """
        Run the full benchmark.

        Returns:
            List of all SolverResult objects (also stored in self.results)
        """
        cfg      = self.config
        all_results = []
        total    = len(cfg.instances) * cfg.n_runs
        done     = 0

        os.makedirs(cfg.output_dir, exist_ok=True)

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Benchmark starting")
            print(f"  Instances : {cfg.instances}")
            print(f"  Runs      : {cfg.n_runs} per solver per instance")
            print(f"{'='*60}\n")

        for inst_name in cfg.instances:
            if verbose:
                print(f"\n--- Instance: {inst_name.upper()} ---")

            try:
                instance = load_instance(inst_name)
            except Exception as e:
                logger.warning(f"Could not load {inst_name}: {e}. Skipping.")
                continue

            for run_idx in range(cfg.n_runs):
                if verbose:
                    print(f"\n  Run {run_idx + 1}/{cfg.n_runs}:")

                solvers = self._build_solvers(run_idx)

                for solver in solvers:
                    solver_name = getattr(solver, "name", type(solver).__name__)
                    if verbose:
                        print(f"    [{solver_name}] ...", end=" ", flush=True)

                    try:
                        t0 = time.time()
                        raw = solver.solve(instance, verbose=False) \
                              if isinstance(solver, NSHybridSolver) \
                              else solver.solve(instance)
                        elapsed = time.time() - t0

                        # NSHybridSolver returns NSResult -- convert to SolverResult
                        if isinstance(solver, NSHybridSolver):
                            from src.benchmark.baselines import SolverResult, _make_result
                            metrics = raw.best_schedule.compute_metrics()
                            result = SolverResult(
                                solver_name    = "NSHybrid",
                                instance_name  = instance.name,
                                makespan       = metrics["makespan"],
                                tardiness      = metrics["tardiness"],
                                utilization    = metrics["utilization"],
                                flowtime       = metrics["flowtime"],
                                energy         = metrics["energy"],
                                fitness        = raw.best_fitness,
                                runtime_sec    = raw.total_time_sec,
                                optimality_gap = raw.optimality_gap,
                            )
                        else:
                            result = raw

                        all_results.append(result)

                        if verbose:
                            gap_str = (f", gap={result.optimality_gap:.1f}%"
                                       if result.optimality_gap is not None else "")
                            print(f"makespan={result.makespan:.0f}{gap_str} "
                                  f"[{elapsed:.1f}s]")

                    except Exception as e:
                        logger.error(f"Solver {solver_name} failed on "
                                     f"{inst_name} run {run_idx}: {e}")
                        if verbose:
                            print(f"FAILED: {e}")

                done += 1
                if verbose:
                    pct = 100 * done / total
                    print(f"\n  Progress: {done}/{total} instance-runs ({pct:.0f}%)")

        self.results = all_results

        if verbose:
            print(f"\n{'='*60}")
            print(f"  Benchmark complete. {len(all_results)} results collected.")
            print(f"{'='*60}")

        return all_results

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        if not self.results:
            raise ValueError("No results yet. Run benchmark first.")
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save(self, path: Optional[str] = None) -> str:
        """Save results to CSV."""
        if not self.results:
            raise ValueError("No results yet. Run benchmark first.")

        if path is None:
            path = os.path.join(
                self.config.output_dir,
                f"benchmark_{int(time.time())}.csv"
            )

        df = self.to_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")
        return path