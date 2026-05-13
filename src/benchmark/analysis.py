"""
Statistical Analysis
====================
Computes statistical significance of benchmark results.

Includes:
  - Summary statistics per solver per instance
  - Wilcoxon signed-rank test (paired, non-parametric)
  - Solver rankings per metric
  - Ablation summary (NS Hybrid vs Pure GA)

Usage:
    from src.benchmark.analysis import BenchmarkAnalyzer
    analyzer = BenchmarkAnalyzer(df)
    analyzer.print_summary()
    analyzer.wilcoxon_test("NS-Hybrid", "PureGA", metric="makespan")
"""

import warnings
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from scipy.stats import wilcoxon


class BenchmarkAnalyzer:
    """
    Statistical analysis of benchmark results DataFrame.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.solvers   = df["solver"].unique().tolist()
        self.instances = df["instance"].unique().tolist()
        self.metrics   = ["makespan", "tardiness", "utilization",
                          "flowtime", "energy", "fitness"]

    # ── Summary statistics ──────────────────────────────────────────────

    def summary(self, metric: str = "makespan") -> pd.DataFrame:
        """
        Mean ± std of a metric per solver per instance.
        Lower is better for all metrics except utilization.
        """
        grouped = (
            self.df.groupby(["solver", "instance"])[metric]
            .agg(["mean", "std", "min", "max", "count"])
            .round(3)
        )
        return grouped

    def summary_all_metrics(self) -> pd.DataFrame:
        """Mean of all metrics per solver, averaged across instances."""
        metrics = [m for m in self.metrics if m != "utilization"]
        agg = self.df.groupby("solver")[metrics].mean().round(3)
        util = self.df.groupby("solver")["utilization"].mean().round(4)
        agg["utilization"] = util
        # Add rank column (by fitness, lower=better)
        agg["rank"] = agg["fitness"].rank().astype(int)
        return agg.sort_values("rank")

    def optimality_gaps(self) -> pd.DataFrame:
        """Mean optimality gap per solver (only for instances with BKS)."""
        has_bks = self.df[self.df["optimality_gap"].notna()]
        if has_bks.empty:
            return pd.DataFrame()
        return (
            has_bks.groupby("solver")["optimality_gap"]
            .agg(["mean", "min", "std"])
            .round(2)
            .sort_values("mean")
        )

    # ── Wilcoxon signed-rank test ───────────────────────────────────────

    def wilcoxon_test(
        self,
        solver_a: str,
        solver_b: str,
        metric: str = "makespan",
        alpha: float = 0.05,
    ) -> dict:
        """
        Wilcoxon signed-rank test: is solver_a significantly better than solver_b?

        Uses paired test across instances (each instance is one pair).
        Non-parametric -- does not assume normal distribution.

        Returns:
            dict with statistic, p_value, significant, better_solver
        """
        # Get mean per instance for each solver
        def _means(solver):
            return (
                self.df[self.df["solver"] == solver]
                .groupby("instance")[metric]
                .mean()
            )

        means_a = _means(solver_a)
        means_b = _means(solver_b)

        # Align on common instances
        common = means_a.index.intersection(means_b.index)
        if len(common) < 3:
            return {
                "solver_a": solver_a,
                "solver_b": solver_b,
                "metric": metric,
                "n_pairs": len(common),
                "statistic": None,
                "p_value": None,
                "significant": False,
                "better_solver": None,
                "note": "Need at least 3 common instances for Wilcoxon test",
            }

        a_vals = means_a[common].values
        b_vals = means_b[common].values
        diff   = a_vals - b_vals

        # Skip if all differences are zero
        if np.all(diff == 0):
            return {
                "solver_a": solver_a, "solver_b": solver_b,
                "metric": metric, "n_pairs": len(common),
                "statistic": 0.0, "p_value": 1.0,
                "significant": False, "better_solver": "tie",
                "note": "All differences are zero",
            }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stat, p_value = wilcoxon(diff, alternative="two-sided")

        # For makespan, tardiness, flowtime, energy, fitness: lower is better
        # For utilization: higher is better
        if metric == "utilization":
            a_better = np.mean(a_vals) > np.mean(b_vals)
        else:
            a_better = np.mean(a_vals) < np.mean(b_vals)

        better = solver_a if a_better else solver_b

        return {
            "solver_a":      solver_a,
            "solver_b":      solver_b,
            "metric":        metric,
            "n_pairs":       len(common),
            "mean_a":        round(float(np.mean(a_vals)), 4),
            "mean_b":        round(float(np.mean(b_vals)), 4),
            "statistic":     round(float(stat), 4),
            "p_value":       round(float(p_value), 6),
            "significant":   bool(p_value < alpha),
            "better_solver": better,
        }

    def wilcoxon_all_pairs(
        self,
        metric: str = "makespan",
        alpha: float = 0.05,
    ) -> pd.DataFrame:
        """Run Wilcoxon test for every solver pair."""
        rows = []
        solvers = self.solvers
        for i in range(len(solvers)):
            for j in range(i + 1, len(solvers)):
                result = self.wilcoxon_test(solvers[i], solvers[j], metric, alpha)
                rows.append(result)
        return pd.DataFrame(rows)

    # ── Ablation ────────────────────────────────────────────────────────

    def ablation_summary(self) -> pd.DataFrame:
        """
        Compare NS Hybrid vs Pure GA -- the core ablation study.
        Shows how much the LLM seeding contributes.
        """
        ns_name = next((s for s in self.solvers if "NS" in s or "Hybrid" in s), None)
        ga_name = next((s for s in self.solvers if s == "PureGA"), None)

        if not ns_name or not ga_name:
            return pd.DataFrame({"note": ["NS Hybrid or PureGA not found in results"]})

        ns_df = self.df[self.df["solver"] == ns_name]
        ga_df = self.df[self.df["solver"] == ga_name]

        rows = []
        for inst in self.instances:
            ns_vals = ns_df[ns_df["instance"] == inst]["makespan"]
            ga_vals = ga_df[ga_df["instance"] == inst]["makespan"]
            if ns_vals.empty or ga_vals.empty:
                continue
            ns_mean = ns_vals.mean()
            ga_mean = ga_vals.mean()
            improvement = 100 * (ga_mean - ns_mean) / ga_mean
            rows.append({
                "instance":            inst,
                f"{ns_name}_makespan": round(ns_mean, 1),
                f"{ga_name}_makespan": round(ga_mean, 1),
                "improvement_%":       round(improvement, 2),
                "ns_better":           ns_mean < ga_mean,
            })

        return pd.DataFrame(rows)

    # ── Printing ─────────────────────────────────────────────────────────

    def print_summary(self):
        """Print a full human-readable benchmark summary."""
        print("\n" + "="*60)
        print("  BENCHMARK RESULTS SUMMARY")
        print("="*60)

        print("\n--- Overall Rankings (by mean fitness, lower=better) ---")
        print(self.summary_all_metrics().to_string())

        print("\n--- Optimality Gaps vs BKS (lower=better) ---")
        gaps = self.optimality_gaps()
        if not gaps.empty:
            print(gaps.to_string())
        else:
            print("  (no BKS available for tested instances)")

        print("\n--- Ablation: NS Hybrid vs Pure GA ---")
        print(self.ablation_summary().to_string(index=False))

        print("\n--- Wilcoxon Test: NS Hybrid vs Pure GA (makespan) ---")
        ns_name = next((s for s in self.solvers if "NS" in s or "Hybrid" in s), None)
        ga_name = next((s for s in self.solvers if s == "PureGA"), None)
        if ns_name and ga_name:
            result = self.wilcoxon_test(ns_name, ga_name, "makespan")
            for k, v in result.items():
                print(f"  {k:<20}: {v}")
        print()
