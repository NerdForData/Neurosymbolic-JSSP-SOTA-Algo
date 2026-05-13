"""
Generate Visualizations
=======================
Run this script to produce all charts and save them to results/plots/.

Usage:
    python generate_plots.py

Output files:
    results/plots/gantt_ft06.png
    results/plots/gantt_ft10.png
    results/plots/convergence_ft06.png
    results/plots/comparison_makespan.png
    results/plots/comparison_fitness.png
    results/plots/radar.png
    results/plots/optimality_gaps.png
"""

import sys
import os
import pandas as pd
sys.path.insert(0, ".")

from src.problem.loader import load_instance
from src.problem.jssp import Schedule, ScheduledOperation
from src.problem.constraints import repair_schedule
from src.llm.llm_client import generate_heuristic_sequences
from src.ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from src.ns_solver import NSHybridSolver
from src.visualization.gantt import plot_gantt
from src.visualization.plots import (
    plot_convergence, plot_comparison,
    plot_radar, plot_optimality_gaps
)

OUTPUT_DIR = "results/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("\n Generating all visualizations...")
    print(f" Output directory: {OUTPUT_DIR}/\n")

    # ── Load instances ───────────────────────────────────────────────
    ft06 = load_instance("ft06")
    ft10 = load_instance("ft10")

    # ── 1. Gantt charts ──────────────────────────────────────────────
    print("[1/5] Gantt charts...")

    for inst, name in [(ft06, "ft06"), (ft10, "ft10")]:
        cfg  = GAConfig(population_size=50, n_generations=100, seed=42)
        seeds = generate_heuristic_sequences(inst, n=5)
        ga   = GeneticAlgorithm(inst, cfg)
        best = ga.evolve(seed_sequences=seeds)

        path = f"{OUTPUT_DIR}/gantt_{name}.png"
        plot_gantt(best.schedule, save_path=path)
        print(f"   Saved: {path}  (makespan={best.schedule.compute_metrics()['makespan']:.0f})")

    # ── 2. Convergence plot ──────────────────────────────────────────
    print("\n[2/5] Convergence plot (NS Hybrid vs Pure GA)...")

    cfg = GAConfig(population_size=50, n_generations=150, seed=42)

    # NS seeded run
    seeds  = generate_heuristic_sequences(ft06, n=5)
    ga_ns  = GeneticAlgorithm(ft06, cfg)
    ga_ns.evolve(seed_sequences=seeds)

    # Pure GA run
    ga_pure = GeneticAlgorithm(ft06, GAConfig(
        population_size=50, n_generations=150, seed=99
    ))
    ga_pure.evolve(seed_sequences=None)

    path = f"{OUTPUT_DIR}/convergence_ft06.png"
    plot_convergence(
        ns_history  = ga_ns.best_per_gen,
        ga_history  = ga_pure.best_per_gen,
        instance_name = "ft06",
        save_path   = path,
    )
    print(f"   Saved: {path}")

    # ── 3. Benchmark comparison charts ──────────────────────────────
    print("\n[3/5] Benchmark comparison charts...")
    print("   Running quick benchmark (this may take ~2 minutes)...")

    from src.benchmark.baseline import (
        DispatchingRuleSolver, PureGASolver, PSOSolver
    )
    from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig

    cfg_bench = BenchmarkConfig(
        instances       = ["ft06", "ft10"],
        n_runs          = 3,
        ga_population   = 30,
        ga_generations  = 80,
        pso_particles   = 20,
        pso_iterations  = 80,
        ortools_time_limit = 15,
        n_llm_seeds     = 3,
        output_dir      = "results",
        run_ortools     = True,
    )
    runner = BenchmarkRunner(cfg_bench)
    runner.run(verbose=True)
    df = runner.to_dataframe()
    csv_path = runner.save("results/benchmark_results.csv")
    print(f"   Results saved: {csv_path}")

    for metric in ["makespan", "fitness", "utilization"]:
        path = f"{OUTPUT_DIR}/comparison_{metric}.png"
        plot_comparison(df, metric=metric, save_path=path)
        print(f"   Saved: {path}")

    # ── 4. Radar chart ───────────────────────────────────────────────
    print("\n[4/5] Radar chart...")
    path = f"{OUTPUT_DIR}/radar.png"
    plot_radar(df, save_path=path)
    print(f"   Saved: {path}")

    # ── 5. Optimality gap chart ──────────────────────────────────────
    print("\n[5/5] Optimality gap chart...")
    path = f"{OUTPUT_DIR}/optimality_gaps.png"
    plot_optimality_gaps(df, save_path=path)
    print(f"   Saved: {path}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f" All charts saved to: {OUTPUT_DIR}/")
    print(f"{'='*50}")
    files = sorted(os.listdir(OUTPUT_DIR))
    for f in files:
        size = os.path.getsize(f"{OUTPUT_DIR}/{f}") // 1024
        print(f"   {f}  ({size} KB)")
    print()

    # Print benchmark summary
    from src.benchmark.analysis import BenchmarkAnalyzer
    analyzer = BenchmarkAnalyzer(df)
    analyzer.print_summary()


if __name__ == "__main__":
    main()