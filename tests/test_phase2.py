"""
Phase 2 Test Suite
==================
Tests for LLM module, GA engine, and NS integration loop.

Run with:  pytest tests/test_phase2.py -v
Note: LLM tests use heuristic fallback when GROQ_API_KEY is not set,
      so all tests pass offline too.
"""

import sys
import random
import pytest
sys.path.insert(0, ".")

from src.problem.loader import load_instance
from src.problem.jssp import Schedule
from src.llm.prompt_builder import build_prompt, build_messages
from src.llm.llm_client import (
    GroqLLMClient, generate_heuristic_sequences,
    _is_valid_sequence, _repair_sequence
)
from src.llm.decoder import decode_sequence, decode_population
from src.ga.genetic_algorithm import (
    GeneticAlgorithm, GAConfig, Individual,
    _ox_crossover, _swap_mutation, _insert_mutation,
    _tournament_select
)
from src.ns_solver import NSHybridSolver, GAConfig


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
def valid_sequence(ft06):
    seq = list(range(ft06.num_jobs)) * ft06.num_machines
    random.shuffle(seq)
    return seq

@pytest.fixture
def heuristic_seqs(ft06):
    return generate_heuristic_sequences(ft06, n=5)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

class TestPromptBuilder:

    def test_build_prompt_returns_string(self, ft06):
        prompt = build_prompt(ft06, n_sequences=3)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_prompt_contains_instance_info(self, ft06):
        prompt = build_prompt(ft06)
        assert "ft06" in prompt
        assert "6" in prompt   # num_jobs / num_machines

    def test_prompt_contains_job_details(self, ft06):
        prompt = build_prompt(ft06)
        assert "Job 0" in prompt
        assert "Job 5" in prompt

    def test_prompt_mentions_objectives(self, ft06):
        prompt = build_prompt(ft06)
        assert "Makespan" in prompt
        assert "Tardiness" in prompt
        assert "Energy" in prompt

    def test_build_messages_format(self, ft06):
        messages = build_messages(ft06, n_sequences=3)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)


# ---------------------------------------------------------------------------
# Sequence validation & repair
# ---------------------------------------------------------------------------

class TestSequenceValidation:

    def test_valid_sequence_passes(self, ft06):
        seq = list(range(ft06.num_jobs)) * ft06.num_machines
        random.shuffle(seq)
        assert _is_valid_sequence(seq, ft06.num_jobs, ft06.num_machines)

    def test_wrong_length_fails(self, ft06):
        seq = list(range(ft06.num_jobs))  # too short
        assert not _is_valid_sequence(seq, ft06.num_jobs, ft06.num_machines)

    def test_invalid_job_id_fails(self, ft06):
        seq = list(range(ft06.num_jobs)) * ft06.num_machines
        seq[0] = 999  # invalid ID
        assert not _is_valid_sequence(seq, ft06.num_jobs, ft06.num_machines)

    def test_repair_produces_valid(self, ft06):
        bad_seq = [0] * (ft06.num_jobs * ft06.num_machines)  # all zeros
        repaired = _repair_sequence(bad_seq, ft06.num_jobs, ft06.num_machines)
        assert _is_valid_sequence(repaired, ft06.num_jobs, ft06.num_machines)

    def test_repair_correct_length(self, ft06):
        bad_seq = list(range(ft06.num_jobs)) * 2  # wrong repeat count
        repaired = _repair_sequence(bad_seq, ft06.num_jobs, ft06.num_machines)
        assert len(repaired) == ft06.num_jobs * ft06.num_machines


# ---------------------------------------------------------------------------
# Heuristic sequences
# ---------------------------------------------------------------------------

class TestHeuristicSequences:

    def test_generates_correct_count(self, ft06):
        seqs = generate_heuristic_sequences(ft06, n=5)
        assert len(seqs) == 5

    def test_all_sequences_valid(self, ft06):
        seqs = generate_heuristic_sequences(ft06, n=8)
        for seq in seqs:
            assert _is_valid_sequence(seq, ft06.num_jobs, ft06.num_machines)

    def test_sequences_are_diverse(self, ft06):
        seqs = generate_heuristic_sequences(ft06, n=4)
        # Not all sequences should be identical
        unique = set(tuple(s) for s in seqs)
        assert len(unique) > 1


# ---------------------------------------------------------------------------
# LLM Client (offline mode)
# ---------------------------------------------------------------------------

class TestLLMClient:

    def test_client_init_without_key(self):
        # Should init without error, fall back to heuristics
        # Patch env to ensure no key is found
        import os
        original = os.environ.pop("GROQ_API_KEY", None)
        try:
            client = GroqLLMClient(api_key="")
            assert not client.is_available
        finally:
            if original:
                os.environ["GROQ_API_KEY"] = original

    def test_fallback_generates_sequences(self, ft06):
        client = GroqLLMClient(api_key="")
        seqs = client.generate_sequences(ft06, n=5)
        assert len(seqs) == 5
        for seq in seqs:
            assert _is_valid_sequence(seq, ft06.num_jobs, ft06.num_machines)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------

class TestDecoder:

    def test_decode_returns_schedule(self, ft06, valid_sequence):
        sched = decode_sequence(valid_sequence, ft06)
        assert isinstance(sched, Schedule)

    def test_decoded_schedule_is_feasible(self, ft06, valid_sequence):
        from src.problem.constraints import ConstraintValidator
        sched = decode_sequence(valid_sequence, ft06)
        validator = ConstraintValidator(ft06)
        assert validator.is_feasible(sched.scheduled_ops)

    def test_decoded_schedule_has_all_ops(self, ft06, valid_sequence):
        sched = decode_sequence(valid_sequence, ft06)
        assert len(sched.scheduled_ops) == ft06.num_operations

    def test_decode_population(self, ft06, heuristic_seqs):
        schedules = decode_population(heuristic_seqs, ft06)
        assert len(schedules) == len(heuristic_seqs)
        for s in schedules:
            assert isinstance(s, Schedule)

    def test_decoded_metrics_computable(self, ft06, valid_sequence):
        sched = decode_sequence(valid_sequence, ft06)
        metrics = sched.compute_metrics()
        assert "makespan" in metrics
        assert metrics["makespan"] > 0
        assert 0 <= metrics["utilization"] <= 1


# ---------------------------------------------------------------------------
# GA operators
# ---------------------------------------------------------------------------

class TestGAOperators:

    def test_ox_crossover_produces_valid(self, ft06, valid_sequence):
        seq_a = valid_sequence
        seq_b = list(range(ft06.num_jobs)) * ft06.num_machines
        random.shuffle(seq_b)
        child_a, child_b = _ox_crossover(
            seq_a, seq_b, ft06.num_jobs, ft06.num_machines
        )
        assert _is_valid_sequence(child_a, ft06.num_jobs, ft06.num_machines)
        assert _is_valid_sequence(child_b, ft06.num_jobs, ft06.num_machines)

    def test_swap_mutation_preserves_validity(self, ft06, valid_sequence):
        mutated = _swap_mutation(valid_sequence)
        assert _is_valid_sequence(mutated, ft06.num_jobs, ft06.num_machines)

    def test_insert_mutation_preserves_validity(self, ft06, valid_sequence):
        mutated = _insert_mutation(valid_sequence)
        assert _is_valid_sequence(mutated, ft06.num_jobs, ft06.num_machines)

    def test_swap_mutation_changes_sequence(self, ft06, valid_sequence):
        # Run multiple swaps — at least one must change the sequence
        # (single swap may pick two identical job IDs, which is valid behaviour)
        changed = any(
            _swap_mutation(valid_sequence) != valid_sequence
            for _ in range(20)
        )
        assert changed

    def test_tournament_select_returns_individual(self, ft06, heuristic_seqs):
        population = []
        for seq in heuristic_seqs:
            ind = Individual(sequence=seq)
            ind.evaluate(ft06)
            population.append(ind)
        winner = _tournament_select(population, tournament_size=3)
        assert isinstance(winner, Individual)
        assert winner.fitness < float("inf")


# ---------------------------------------------------------------------------
# Genetic Algorithm
# ---------------------------------------------------------------------------

class TestGeneticAlgorithm:

    def test_ga_runs_and_returns_individual(self, ft06, heuristic_seqs):
        cfg = GAConfig(population_size=10, n_generations=5, seed=42)
        ga  = GeneticAlgorithm(ft06, cfg)
        best = ga.evolve(seed_sequences=heuristic_seqs)
        assert isinstance(best, Individual)
        assert best.fitness < float("inf")

    def test_ga_fitness_improves_or_holds(self, ft06, heuristic_seqs):
        cfg = GAConfig(population_size=20, n_generations=20, seed=42)
        ga  = GeneticAlgorithm(ft06, cfg)
        ga.evolve(seed_sequences=heuristic_seqs)
        # Best fitness should never increase generation to generation
        for i in range(1, len(ga.best_per_gen)):
            assert ga.best_per_gen[i] <= ga.best_per_gen[i-1] + 1e-9

    def test_ga_convergence_stats(self, ft06, heuristic_seqs):
        cfg = GAConfig(population_size=10, n_generations=10, seed=42)
        ga  = GeneticAlgorithm(ft06, cfg)
        ga.evolve(seed_sequences=heuristic_seqs)
        stats = ga.convergence_stats()
        assert len(stats["best_per_gen"]) == 10
        assert len(stats["avg_per_gen"])  == 10

    def test_ga_no_seeds_still_works(self, ft06):
        cfg = GAConfig(population_size=10, n_generations=5, seed=42)
        ga  = GeneticAlgorithm(ft06, cfg)
        best = ga.evolve(seed_sequences=None)
        assert best.fitness < float("inf")

    def test_best_schedule_is_feasible(self, ft06, heuristic_seqs):
        from src.problem.constraints import ConstraintValidator
        cfg = GAConfig(population_size=10, n_generations=10, seed=42)
        ga  = GeneticAlgorithm(ft06, cfg)
        best = ga.evolve(seed_sequences=heuristic_seqs)
        validator = ConstraintValidator(ft06)
        assert validator.is_feasible(best.schedule.scheduled_ops)


# ---------------------------------------------------------------------------
# NS Hybrid Solver (end-to-end)
# ---------------------------------------------------------------------------

class TestNSSolver:

    def test_solver_returns_result(self, ft06):
        cfg    = GAConfig(population_size=10, n_generations=10, seed=42)
        solver = NSHybridSolver(ga_config=cfg, n_llm_seeds=3)
        result = solver.solve(ft06, verbose=False)
        assert result is not None
        assert result.best_makespan > 0

    def test_solver_result_is_feasible(self, ft06):
        from src.problem.constraints import ConstraintValidator
        cfg    = GAConfig(population_size=10, n_generations=10, seed=42)
        solver = NSHybridSolver(ga_config=cfg, n_llm_seeds=3)
        result = solver.solve(ft06, verbose=False)
        validator = ConstraintValidator(ft06)
        assert validator.is_feasible(result.best_schedule.scheduled_ops)

    def test_solver_better_than_heuristic_baseline(self, ft06):
        """GA+LLM should outperform plain heuristic after even 50 generations."""
        from src.problem.constraints import ConstraintValidator
        from src.llm.llm_client import generate_heuristic_sequences
        from src.llm.decoder import decode_population

        # Heuristic baseline
        seqs      = generate_heuristic_sequences(ft06, n=5)
        schedules = decode_population(seqs, ft06)
        baseline  = min(s.fitness() for s in schedules)

        # NS Hybrid
        cfg    = GAConfig(population_size=20, n_generations=50, seed=42)
        solver = NSHybridSolver(ga_config=cfg, n_llm_seeds=5)
        result = solver.solve(ft06, verbose=False)

        assert result.best_fitness <= baseline

    def test_solver_summary_string(self, ft06):
        cfg    = GAConfig(population_size=10, n_generations=5, seed=42)
        solver = NSHybridSolver(ga_config=cfg, n_llm_seeds=3)
        result = solver.solve(ft06, verbose=False)
        summary = result.summary()
        assert "Makespan" in summary
        assert "Fitness"  in summary