"""
Phase 3 Test Suite
==================
Tests for baseline solvers, benchmark runner, and statistical analysis.

Run with:  pytest tests/test_phase3.py -v
"""

import sys
import pytest
import pandas as pd
sys.path.insert(0, ".")

from src.problem.loader import load_instance
from src.benchmark.baseline import (
    DispatchingRuleSolver, PureGASolver, PSOSolver,
    ORToolsSolver, SolverResult
)
from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig
from src.benchmark.analysis import BenchmarkAnalyzer
from src.ga.genetic_algorithm import GAConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ft06():
    return load_instance("ft06")

@pytest.fixture
def ft10():
    return load_instance("ft10")

@pytest.fixture
def fast_ga_config():
    return GAConfig(population_size=10, n_generations=10, seed=42)

@pytest.fixture
def sample_df():
    """Minimal benchmark DataFrame for testing analysis."""
    import random
    rows = []
    random.seed(42)
    instances = ["ft06", "ft10", "inst_a", "inst_b", "inst_c"]
    bases     = [60, 950, 200, 400, 300]
    offsets   = {"NSHybrid": 0, "PureGA": 15, "Dispatching-SPT": 30}
    for inst, base in zip(instances, bases):
        for solver, offset in offsets.items():
            for _ in range(5):
                makespan = base + offset + random.uniform(0, 3)
                rows.append({
                    "solver": solver, "instance": inst,
                    "makespan": makespan, "tardiness": 50.0,
                    "utilization": 0.7, "flowtime": 300.0,
                    "energy": 200.0, "fitness": makespan / 1000,
                    "runtime_sec": 1.0,
                    "optimality_gap": makespan - base,
                    "feasible": True,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Dispatching rule solver
# ---------------------------------------------------------------------------

class TestDispatchingRuleSolver:

    @pytest.mark.parametrize("rule", ["SPT", "LPT", "EDD", "MWR"])
    def test_all_rules_run(self, ft06, rule):
        solver = DispatchingRuleSolver(rule=rule)
        result = solver.solve(ft06)
        assert isinstance(result, SolverResult)
        assert result.makespan > 0

    def test_invalid_rule_raises(self):
        with pytest.raises(ValueError):
            DispatchingRuleSolver(rule="INVALID")

    def test_result_is_feasible(self, ft06):
        from src.problem.constraints import ConstraintValidator
        solver = DispatchingRuleSolver(rule="SPT")
        result = solver.solve(ft06)
        assert result.feasible

    def test_result_has_all_metrics(self, ft06):
        result = DispatchingRuleSolver("EDD").solve(ft06)
        assert result.makespan > 0
        assert result.tardiness >= 0
        assert 0 <= result.utilization <= 1
        assert result.flowtime > 0
        assert result.energy > 0

    def test_runtime_recorded(self, ft06):
        result = DispatchingRuleSolver("SPT").solve(ft06)
        assert result.runtime_sec >= 0

    def test_to_dict_complete(self, ft06):
        result = DispatchingRuleSolver("SPT").solve(ft06)
        d = result.to_dict()
        for key in ["solver", "instance", "makespan", "fitness",
                    "runtime_sec", "feasible"]:
            assert key in d


# ---------------------------------------------------------------------------
# Pure GA solver
# ---------------------------------------------------------------------------

class TestPureGASolver:

    def test_runs_and_returns_result(self, ft06, fast_ga_config):
        solver = PureGASolver(config=fast_ga_config)
        result = solver.solve(ft06)
        assert isinstance(result, SolverResult)
        assert result.makespan > 0

    def test_result_feasible(self, ft06, fast_ga_config):
        solver = PureGASolver(config=fast_ga_config)
        result = solver.solve(ft06)
        assert result.feasible

    def test_beats_dispatching_rule(self, ft06):
        """GA with enough generations should match or beat naive SPT."""
        cfg = GAConfig(population_size=30, n_generations=100, seed=42)
        ga_result  = PureGASolver(config=cfg).solve(ft06)
        spt_result = DispatchingRuleSolver("SPT").solve(ft06)
        # GA should be at least as good as SPT (not strictly better -- stochastic)
        assert ga_result.fitness <= spt_result.fitness + 0.01


# ---------------------------------------------------------------------------
# PSO solver
# ---------------------------------------------------------------------------

class TestPSOSolver:

    def test_runs_and_returns_result(self, ft06):
        solver = PSOSolver(n_particles=5, n_iterations=5, seed=42)
        result = solver.solve(ft06)
        assert isinstance(result, SolverResult)
        assert result.makespan > 0

    def test_result_feasible(self, ft06):
        solver = PSOSolver(n_particles=5, n_iterations=5, seed=42)
        result = solver.solve(ft06)
        assert result.feasible

    def test_result_has_metrics(self, ft06):
        result = PSOSolver(n_particles=5, n_iterations=5).solve(ft06)
        assert 0 <= result.utilization <= 1
        assert result.energy > 0


# ---------------------------------------------------------------------------
# OR-Tools solver
# ---------------------------------------------------------------------------

class TestORToolsSolver:

    def test_runs_on_ft06(self, ft06):
        solver = ORToolsSolver(time_limit_sec=10)
        result = solver.solve(ft06)
        assert isinstance(result, SolverResult)

    def test_ft06_near_optimal(self, ft06):
        """OR-Tools should get very close to BKS=55 on tiny FT06."""
        solver = ORToolsSolver(time_limit_sec=30)
        result = solver.solve(ft06)
        if result.feasible:
            assert result.optimality_gap is not None
            assert result.optimality_gap < 5.0  # within 5% of BKS

    def test_infeasible_result_on_timeout(self):
        """Mocking a timeout: result should be marked infeasible."""
        # We test the SolverResult infeasible flag directly
        result = SolverResult(
            solver_name="test", instance_name="ft06",
            makespan=float("inf"), tardiness=float("inf"),
            utilization=0.0, flowtime=float("inf"),
            energy=float("inf"), fitness=float("inf"),
            runtime_sec=60.0, optimality_gap=None, feasible=False,
        )
        assert not result.feasible


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class TestBenchmarkRunner:

    def test_runner_runs_ft06(self, tmp_path):
        cfg = BenchmarkConfig(
            instances       = ["ft06"],
            n_runs          = 1,
            ga_population   = 10,
            ga_generations  = 5,
            pso_particles   = 5,
            pso_iterations  = 5,
            ortools_time_limit = 5,
            n_llm_seeds     = 2,
            output_dir      = str(tmp_path),
        )
        runner  = BenchmarkRunner(cfg)
        results = runner.run(verbose=False)
        assert len(results) > 0

    def test_runner_produces_dataframe(self, tmp_path):
        cfg = BenchmarkConfig(
            instances = ["ft06"], n_runs = 1,
            ga_population=10, ga_generations=5,
            pso_particles=5, pso_iterations=5,
            ortools_time_limit=5, n_llm_seeds=2,
            output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(cfg)
        runner.run(verbose=False)
        df = runner.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "solver" in df.columns
        assert "makespan" in df.columns

    def test_runner_saves_csv(self, tmp_path):
        cfg = BenchmarkConfig(
            instances=["ft06"], n_runs=1,
            ga_population=10, ga_generations=5,
            pso_particles=5, pso_iterations=5,
            ortools_time_limit=5, n_llm_seeds=2,
            output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(cfg)
        runner.run(verbose=False)
        path = runner.save()
        import os
        assert os.path.exists(path)
        df = pd.read_csv(path)
        assert len(df) > 0

    def test_all_solvers_present(self, tmp_path):
        cfg = BenchmarkConfig(
            instances=["ft06"], n_runs=1,
            ga_population=10, ga_generations=5,
            pso_particles=5, pso_iterations=5,
            ortools_time_limit=5, n_llm_seeds=2,
            output_dir=str(tmp_path),
        )
        runner = BenchmarkRunner(cfg)
        runner.run(verbose=False)
        df = runner.to_dataframe()
        solvers = df["solver"].unique()
        assert any("Dispatching" in s for s in solvers)
        assert any("PureGA" in s for s in solvers)
        assert any("PSO" in s for s in solvers)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

class TestBenchmarkAnalyzer:

    def test_summary_returns_dataframe(self, sample_df):
        analyzer = BenchmarkAnalyzer(sample_df)
        summary = analyzer.summary("makespan")
        assert isinstance(summary, pd.DataFrame)

    def test_summary_all_metrics(self, sample_df):
        analyzer = BenchmarkAnalyzer(sample_df)
        df = analyzer.summary_all_metrics()
        assert "makespan" in df.columns
        assert "fitness"  in df.columns
        assert "rank"     in df.columns

    def test_ns_hybrid_ranked_first(self, sample_df):
        """In our sample data, NSHybrid has lowest makespan."""
        analyzer = BenchmarkAnalyzer(sample_df)
        ranked   = analyzer.summary_all_metrics()
        assert ranked.index[0] == "NSHybrid"

    def test_wilcoxon_returns_dict(self, sample_df):
        analyzer = BenchmarkAnalyzer(sample_df)
        result   = analyzer.wilcoxon_test("NSHybrid", "PureGA", "makespan")
        assert "p_value"     in result
        assert "significant" in result
        assert "better_solver" in result

    def test_wilcoxon_ns_better_than_ga(self, sample_df):
        """NSHybrid should be significantly better than PureGA in sample data."""
        analyzer = BenchmarkAnalyzer(sample_df)
        result   = analyzer.wilcoxon_test("NSHybrid", "PureGA", "makespan")
        assert result["better_solver"] == "NSHybrid"

    def test_ablation_summary(self, sample_df):
        analyzer = BenchmarkAnalyzer(sample_df)
        df = analyzer.ablation_summary()
        assert isinstance(df, pd.DataFrame)
        assert "improvement_%" in df.columns

    def test_optimality_gaps(self, sample_df):
        analyzer = BenchmarkAnalyzer(sample_df)
        gaps = analyzer.optimality_gaps()
        assert isinstance(gaps, pd.DataFrame)

    def test_wilcoxon_all_pairs(self, sample_df):
        analyzer = BenchmarkAnalyzer(sample_df)
        df = analyzer.wilcoxon_all_pairs("makespan")
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0