"""
Baseline Solvers
================
All competitor algorithms for benchmarking against the NS Hybrid.

Solvers implemented:
  1. DispatchingRuleSolver  -- SPT, LPT, EDD (industrial baseline)
  2. PureGASolver           -- GA without LLM seeding (ablation study)
  3. PSOSolver              -- Particle Swarm Optimisation
  4. ORToolsSolver          -- Exact CP-SAT solver (quality upper bound)

All solvers share the same interface:
    solver.solve(instance) -> SolverResult
"""

import time
import random
import logging
from dataclasses import dataclass
from typing import Optional, List

from src.problem.jssp import JSSPInstance, Schedule
from src.llm.decoder import decode_sequence
from src.ga.genetic_algorithm import GeneticAlgorithm, GAConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared result container
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    """Unified result format for all solvers."""
    solver_name:    str
    instance_name:  str
    makespan:       float
    tardiness:      float
    utilization:    float
    flowtime:       float
    energy:         float
    fitness:        float
    runtime_sec:    float
    optimality_gap: Optional[float]
    feasible:       bool = True

    def to_dict(self) -> dict:
        return {
            "solver":          self.solver_name,
            "instance":        self.instance_name,
            "makespan":        round(self.makespan, 2),
            "tardiness":       round(self.tardiness, 2),
            "utilization":     round(self.utilization, 4),
            "flowtime":        round(self.flowtime, 2),
            "energy":          round(self.energy, 2),
            "fitness":         round(self.fitness, 6),
            "runtime_sec":     round(self.runtime_sec, 3),
            "optimality_gap":  round(self.optimality_gap, 2) if self.optimality_gap is not None else None,
            "feasible":        self.feasible,
        }


def _make_result(
    solver_name: str,
    instance: JSSPInstance,
    schedule: Schedule,
    runtime: float,
) -> SolverResult:
    """Helper to build a SolverResult from a Schedule."""
    metrics = schedule.compute_metrics()
    return SolverResult(
        solver_name    = solver_name,
        instance_name  = instance.name,
        makespan       = metrics["makespan"],
        tardiness      = metrics["tardiness"],
        utilization    = metrics["utilization"],
        flowtime       = metrics["flowtime"],
        energy         = metrics["energy"],
        fitness        = schedule.fitness(),
        runtime_sec    = runtime,
        optimality_gap = schedule.optimality_gap(),
    )


# ---------------------------------------------------------------------------
# 1. Dispatching Rule Solver
# ---------------------------------------------------------------------------

class DispatchingRuleSolver:
    """
    Classic priority dispatching rules.
    These are what most real factories use today -- our industrial baseline.

    Rules:
      SPT -- Shortest Processing Time first (minimises average flow time)
      LPT -- Longest Processing Time first (maximises machine utilisation)
      EDD -- Earliest Due Date first (minimises tardiness)
      MWR -- Most Work Remaining (balances load across machines)
    """

    RULES = ["SPT", "LPT", "EDD", "MWR"]

    def __init__(self, rule: str = "SPT"):
        rule = rule.upper()
        if rule not in self.RULES:
            raise ValueError(f"Unknown rule '{rule}'. Choose from {self.RULES}")
        self.rule = rule
        self.name = f"Dispatching-{rule}"

    def _build_sequence(self, instance: JSSPInstance) -> List[int]:
        """Build a priority sequence based on the dispatching rule."""
        if self.rule == "SPT":
            jobs = sorted(instance.jobs, key=lambda j: j.total_processing_time)
        elif self.rule == "LPT":
            jobs = sorted(instance.jobs,
                          key=lambda j: j.total_processing_time, reverse=True)
        elif self.rule == "EDD":
            jobs = sorted(instance.jobs,
                          key=lambda j: (j.due_date or float("inf")))
        elif self.rule == "MWR":
            jobs = sorted(instance.jobs,
                          key=lambda j: j.total_processing_time, reverse=True)

        seq = []
        for _ in range(instance.num_machines):
            seq.extend(j.job_id for j in jobs)
        return seq

    def solve(self, instance: JSSPInstance) -> SolverResult:
        start = time.time()
        seq      = self._build_sequence(instance)
        schedule = decode_sequence(seq, instance)
        return _make_result(self.name, instance, schedule, time.time() - start)


# ---------------------------------------------------------------------------
# 2. Pure GA Solver (ablation -- no LLM seeding)
# ---------------------------------------------------------------------------

class PureGASolver:
    """
    Standard GA with random initialisation -- no LLM seeding.
    This is the ablation baseline that proves LLM seeding adds value.
    Direct comparison: PureGA vs NSHybrid = the neuro-symbolic contribution.
    """

    def __init__(self, config: Optional[GAConfig] = None):
        self.config = config or GAConfig()
        self.name   = "PureGA"

    def solve(self, instance: JSSPInstance) -> SolverResult:
        start = time.time()
        ga    = GeneticAlgorithm(instance, self.config)
        best  = ga.evolve(seed_sequences=None)   # no LLM seeds
        return _make_result(self.name, instance, best.schedule, time.time() - start)


# ---------------------------------------------------------------------------
# 3. PSO Solver
# ---------------------------------------------------------------------------

class PSOSolver:
    """
    Particle Swarm Optimisation for JSSP.

    Representation: each particle is a continuous velocity vector
    that gets discretised into a priority sequence via ranking.
    This is the standard DABS (Discrete Particle Swarm) encoding.
    """

    def __init__(
        self,
        n_particles:  int = 30,
        n_iterations: int = 200,
        w:  float = 0.7,    # inertia weight
        c1: float = 1.5,    # cognitive coefficient
        c2: float = 1.5,    # social coefficient
        seed: Optional[int] = 42,
    ):
        self.n_particles  = n_particles
        self.n_iterations = n_iterations
        self.w  = w
        self.c1 = c1
        self.c2 = c2
        self.seed = seed
        self.name = "PSO"

    def _vec_to_sequence(self, vec: List[float], instance: JSSPInstance) -> List[int]:
        """
        Convert continuous PSO vector to a valid priority sequence.
        Uses the Smallest Position Value (SPV) rule.
        """
        n_jobs  = instance.num_jobs
        n_mach  = instance.num_machines
        # Split vector into n_machines chunks, one per operation slot
        indexed = sorted(range(len(vec)), key=lambda i: vec[i])
        # Map back to job IDs using modulo
        seq = [i % n_jobs for i in indexed]
        # Repair to valid sequence
        from src.llm.llm_client import _repair_sequence
        return _repair_sequence(seq, n_jobs, n_mach)

    def solve(self, instance: JSSPInstance) -> SolverResult:
        if self.seed:
            random.seed(self.seed)

        start   = time.time()
        n       = instance.num_jobs * instance.num_machines
        n_jobs  = instance.num_jobs
        n_mach  = instance.num_machines

        # Initialise particles
        positions  = [[random.uniform(0, n_jobs) for _ in range(n)]
                      for _ in range(self.n_particles)]
        velocities = [[random.uniform(-1, 1) for _ in range(n)]
                      for _ in range(self.n_particles)]

        # Evaluate initial positions
        def _fitness(pos):
            seq  = self._vec_to_sequence(pos, instance)
            sched = decode_sequence(seq, instance)
            return sched.fitness(), sched

        personal_best_pos = [p[:] for p in positions]
        personal_best_fit = []
        personal_best_sch = []
        for pos in positions:
            f, s = _fitness(pos)
            personal_best_fit.append(f)
            personal_best_sch.append(s)

        global_best_idx = personal_best_fit.index(min(personal_best_fit))
        global_best_pos = personal_best_pos[global_best_idx][:]
        global_best_fit = personal_best_fit[global_best_idx]
        global_best_sch = personal_best_sch[global_best_idx]

        # Main PSO loop
        for _ in range(self.n_iterations):
            for i in range(self.n_particles):
                r1 = [random.random() for _ in range(n)]
                r2 = [random.random() for _ in range(n)]

                # Update velocity
                velocities[i] = [
                    self.w  * velocities[i][d]
                    + self.c1 * r1[d] * (personal_best_pos[i][d] - positions[i][d])
                    + self.c2 * r2[d] * (global_best_pos[d]       - positions[i][d])
                    for d in range(n)
                ]

                # Update position
                positions[i] = [
                    positions[i][d] + velocities[i][d]
                    for d in range(n)
                ]

                # Evaluate
                f, s = _fitness(positions[i])
                if f < personal_best_fit[i]:
                    personal_best_fit[i] = f
                    personal_best_pos[i] = positions[i][:]
                    personal_best_sch[i] = s
                    if f < global_best_fit:
                        global_best_fit = f
                        global_best_pos = positions[i][:]
                        global_best_sch = s

        return _make_result(self.name, instance, global_best_sch, time.time() - start)


# ---------------------------------------------------------------------------
# 4. OR-Tools CP-SAT Solver (exact -- quality upper bound)
# ---------------------------------------------------------------------------

class ORToolsSolver:
    """
    Google OR-Tools CP-SAT exact solver.
    Provides the best-possible solution (or near-optimal within time limit).
    Used as quality upper bound in benchmarking.
    Will time out on large instances (Tier 3) -- that's expected and reported.
    """

    def __init__(self, time_limit_sec: int = 60):
        self.time_limit = time_limit_sec
        self.name       = f"OR-Tools(t={time_limit_sec}s)"

    def solve(self, instance: JSSPInstance) -> SolverResult:
        try:
            from ortools.sat.python import cp_model
        except ImportError:
            raise ImportError("Run: pip install ortools")

        start = time.time()
        model = cp_model.CpModel()

        # Compute horizon (upper bound on makespan)
        horizon = sum(
            op.processing_time
            for job in instance.jobs
            for op in job.operations
        )

        # Decision variables: (start, end, interval) per operation
        all_tasks = {}
        machine_intervals = {m: [] for m in range(instance.num_machines)}

        for job in instance.jobs:
            for op in job.operations:
                suffix = f"_j{op.job_id}_s{op.step}"
                start_var    = model.NewIntVar(0, horizon, f"start{suffix}")
                end_var      = model.NewIntVar(0, horizon, f"end{suffix}")
                interval_var = model.NewIntervalVar(
                    start_var, op.processing_time, end_var, f"interval{suffix}"
                )
                all_tasks[(op.job_id, op.step)] = (start_var, end_var, interval_var)
                machine_intervals[op.machine_id].append(interval_var)

        # Constraint 1: no overlap on machines
        for m in range(instance.num_machines):
            model.AddNoOverlap(machine_intervals[m])

        # Constraint 2: job precedence
        for job in instance.jobs:
            for step in range(len(job.operations) - 1):
                _, end_prev, _   = all_tasks[(job.job_id, step)]
                start_next, _, _ = all_tasks[(job.job_id, step + 1)]
                model.Add(start_next >= end_prev)

        # Objective: minimise makespan
        makespan_var = model.NewIntVar(0, horizon, "makespan")
        model.AddMaxEquality(makespan_var, [
            all_tasks[(job.job_id, job.num_operations - 1)][1]
            for job in instance.jobs
        ])
        model.Minimize(makespan_var)

        # Solve
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers  = 4
        status = solver.Solve(model)

        runtime = time.time() - start

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            # Build schedule from solution
            from src.problem.jssp import ScheduledOperation
            scheduled_ops = []
            for job in instance.jobs:
                for op in job.operations:
                    s, e, _ = all_tasks[(op.job_id, op.step)]
                    scheduled_ops.append(ScheduledOperation(
                        operation  = op,
                        start_time = solver.Value(s),
                        end_time   = solver.Value(e),
                    ))
            from src.problem.jssp import Schedule
            schedule = Schedule(instance=instance, scheduled_ops=scheduled_ops)
            result = _make_result(self.name, instance, schedule, runtime)
            if status == cp_model.FEASIBLE:
                logger.info(f"OR-Tools: feasible (not proven optimal, timed out)")
            return result
        else:
            logger.warning(f"OR-Tools: no solution found within {self.time_limit}s")
            # Return a dummy infeasible result
            return SolverResult(
                solver_name    = self.name,
                instance_name  = instance.name,
                makespan       = float("inf"),
                tardiness      = float("inf"),
                utilization    = 0.0,
                flowtime       = float("inf"),
                energy         = float("inf"),
                fitness        = float("inf"),
                runtime_sec    = runtime,
                optimality_gap = None,
                feasible       = False,
            )