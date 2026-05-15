"""
Phase 4 Test Suite
==================
Tests for Gantt chart and benchmark visualization.

Run with:  pytest tests/test_phase4.py -v
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")

from src.problem.loader import load_instance
from src.problem.constraints import repair_schedule
from src.problem.jssp import Schedule, ScheduledOperation
from src.llm.llm_client import generate_heuristic_sequences
from src.llm.decoder import decode_sequence
from src.ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from src.visualization.gantt import plot_gantt
from src.visualization.plots import (
    plot_convergence, plot_comparison,
    plot_radar, plot_optimality_gaps
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ft06():
    return load_instance("ft06")

@pytest.fixture
def ft06_schedule(ft06):
    """A valid repaired schedule for FT06."""
    dummy = []
    for job in ft06.jobs:
        for op in job.operations:
            dummy.append(ScheduledOperation(
                operation=op, start_time=0, end_time=op.processing_time
            ))
    repaired = repair_schedule(dummy, ft06)
    return Schedule(instance=ft06, scheduled_ops=repaired)

@pytest.fixture
def ga_histories(ft06):
    """Run two GAs (NS seeded and pure random) and return their histories."""
    cfg = GAConfig(population_size=15, n_generations=20, seed=42)

    # NS seeded
    seeds = generate_heuristic_sequences(ft06, n=5)
    ga_ns = GeneticAlgorithm(ft06, cfg)
    ga_ns.evolve(seed_sequences=seeds)

    # Pure GA
    ga_pure = GeneticAlgorithm(ft06, GAConfig(
        population_size=15, n_generations=20, seed=99
    ))
    ga_pure.evolve(seed_sequences=None)

    return ga_ns.best_per_gen, ga_pure.best_per_gen

@pytest.fixture
def sample_df():
    """Sample benchmark results DataFrame."""
    import random
    random.seed(42)
    rows = []
    for inst, base in [("ft06", 60), ("ft10", 950)]:
        for solver, offset in [("NSHybrid", 0), ("PureGA", 15),
                                ("Dispatching-SPT", 25), ("PSO", 10)]:
            for _ in range(3):
                makespan = base + offset + random.uniform(0, 3)
                rows.append({
                    "solver": solver, "instance": inst,
                    "makespan": makespan, "tardiness": 50.0,
                    "utilization": 0.65 + random.uniform(0, 0.1),
                    "flowtime": 300.0, "energy": 200.0,
                    "fitness": makespan / 1000,
                    "runtime_sec": 1.0,
                    "optimality_gap": makespan - base,
                    "feasible": True,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Gantt chart
# ---------------------------------------------------------------------------

class TestGanttChart:

    def test_returns_figure(self, ft06_schedule):
        fig = plot_gantt(ft06_schedule)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_figure_has_axes(self, ft06_schedule):
        fig = plot_gantt(ft06_schedule)
        assert len(fig.axes) >= 1
        plt.close(fig)

    def test_saves_png(self, ft06_schedule, tmp_path):
        path = str(tmp_path / "gantt_test.png")
        plot_gantt(ft06_schedule, save_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
        plt.close("all")

    def test_saves_pdf(self, ft06_schedule, tmp_path):
        path = str(tmp_path / "gantt_test.pdf")
        plot_gantt(ft06_schedule, save_path=path)
        assert os.path.exists(path)
        plt.close("all")

    def test_custom_title(self, ft06_schedule):
        fig = plot_gantt(ft06_schedule, title="My Custom Title")
        ax = fig.axes[0]
        assert "My Custom Title" in ax.get_title()
        plt.close(fig)

    def test_all_operations_plotted(self, ft06_schedule):
        """Number of patches should match number of operations."""
        fig = plot_gantt(ft06_schedule)
        ax = fig.axes[0]
        # FancyBboxPatch count should equal number of operations
        n_patches = sum(
            1 for p in ax.patches
            if isinstance(p, plt.matplotlib.patches.FancyBboxPatch)
        )
        assert n_patches == ft06_schedule.instance.num_operations
        plt.close(fig)

    def test_works_with_optimized_schedule(self, ft06):
        """Test with a GA-optimized schedule, not just repaired."""
        cfg  = GAConfig(population_size=10, n_generations=10, seed=42)
        ga   = GeneticAlgorithm(ft06, cfg)
        best = ga.evolve()
        fig  = plot_gantt(best.schedule)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

class TestConvergencePlot:

    def test_returns_figure(self, ga_histories):
        ns_hist, ga_hist = ga_histories
        fig = plot_convergence(ns_hist, ga_hist, instance_name="ft06")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, ga_histories, tmp_path):
        ns_hist, ga_hist = ga_histories
        path = str(tmp_path / "convergence.png")
        plot_convergence(ns_hist, ga_hist, save_path=path)
        assert os.path.exists(path)
        plt.close("all")

    def test_has_two_lines(self, ga_histories):
        ns_hist, ga_hist = ga_histories
        fig = plot_convergence(ns_hist, ga_hist)
        ax = fig.axes[0]
        lines = ax.get_lines()
        assert len(lines) >= 2
        plt.close(fig)

    def test_axis_labels(self, ga_histories):
        ns_hist, ga_hist = ga_histories
        fig = plot_convergence(ns_hist, ga_hist)
        ax = fig.axes[0]
        assert "Generation" in ax.get_xlabel()
        assert "Fitness" in ax.get_ylabel()
        plt.close(fig)


# ---------------------------------------------------------------------------
# Comparison bar chart
# ---------------------------------------------------------------------------

class TestComparisonPlot:

    def test_returns_figure(self, sample_df):
        fig = plot_comparison(sample_df, metric="makespan")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, sample_df, tmp_path):
        path = str(tmp_path / "comparison.png")
        plot_comparison(sample_df, save_path=path)
        assert os.path.exists(path)
        plt.close("all")

    def test_filtered_by_instance(self, sample_df):
        fig = plot_comparison(sample_df, metric="makespan", instance="ft06")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_all_metrics_work(self, sample_df):
        for metric in ["makespan", "tardiness", "utilization",
                       "flowtime", "energy", "fitness"]:
            fig = plot_comparison(sample_df, metric=metric)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)


# ---------------------------------------------------------------------------
# Radar chart
# ---------------------------------------------------------------------------

class TestRadarPlot:

    def test_returns_figure(self, sample_df):
        fig = plot_radar(sample_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, sample_df, tmp_path):
        path = str(tmp_path / "radar.png")
        plot_radar(sample_df, save_path=path)
        assert os.path.exists(path)
        plt.close("all")

    def test_subset_of_solvers(self, sample_df):
        fig = plot_radar(sample_df, solvers=["NSHybrid", "PureGA"])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Optimality gap plot
# ---------------------------------------------------------------------------

class TestOptimalityGapPlot:

    def test_returns_figure(self, sample_df):
        fig = plot_optimality_gaps(sample_df)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_saves_file(self, sample_df, tmp_path):
        path = str(tmp_path / "gaps.png")
        plot_optimality_gaps(sample_df, save_path=path)
        assert os.path.exists(path)
        plt.close("all")

    def test_empty_df_handled(self, sample_df):
        """Should not crash if no BKS data available."""
        df_no_bks = sample_df.copy()
        df_no_bks["optimality_gap"] = None
        fig = plot_optimality_gaps(df_no_bks)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)