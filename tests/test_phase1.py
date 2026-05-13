"""
Phase 1 Test Suite
==================
Run with:  pytest tests/test_phase1.py -v
"""

import sys
import pytest
sys.path.insert(0, ".")

from src.problem.jssp import (
    Operation, Job, Machine, JSSPInstance,
    ScheduledOperation, Schedule
)
from src.problem.loader import load_instance, BEST_KNOWN
from src.problem.constraints import ConstraintValidator, repair_schedule


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
def dummy_infeasible(ft06):
    """All operations start at time 0 -- guaranteed violations."""
    ops = []
    for job in ft06.jobs:
        for op in job.operations:
            ops.append(ScheduledOperation(
                operation=op, start_time=0, end_time=op.processing_time
            ))
    return ops

@pytest.fixture
def repaired_schedule(ft06, dummy_infeasible):
    return repair_schedule(dummy_infeasible, ft06)


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------

class TestLoader:

    def test_ft06_loads(self, ft06):
        assert ft06 is not None
        assert ft06.name == "ft06"

    def test_ft06_dimensions(self, ft06):
        assert ft06.num_jobs == 6
        assert ft06.num_machines == 6
        assert ft06.num_operations == 36

    def test_ft10_dimensions(self, ft10):
        assert ft10.num_jobs == 10
        assert ft10.num_machines == 10
        assert ft10.num_operations == 100

    def test_bks_loaded(self, ft06):
        assert ft06.best_known_solution == 55

    def test_due_dates_assigned(self, ft06):
        for job in ft06.jobs:
            assert job.due_date is not None
            assert job.due_date > 0

    def test_operation_structure(self, ft06):
        for job in ft06.jobs:
            for step, op in enumerate(job.operations):
                assert op.job_id == job.job_id
                assert op.step == step
                assert 0 <= op.machine_id < ft06.num_machines
                assert op.processing_time > 0

    def test_objective_weights_sum_to_one(self, ft06):
        total = (ft06.w_makespan + ft06.w_tardiness +
                 ft06.w_utilization + ft06.w_flowtime + ft06.w_energy)
        assert abs(total - 1.0) < 1e-6

    def test_unknown_instance_raises(self):
        with pytest.raises((FileNotFoundError, ConnectionError, ValueError)):
            load_instance("nonexistent_xyz")


# ---------------------------------------------------------------------------
# Constraint validation
# ---------------------------------------------------------------------------

class TestConstraintValidator:

    def test_infeasible_schedule_has_violations(self, ft06, dummy_infeasible):
        validator = ConstraintValidator(ft06)
        violations = validator.validate(dummy_infeasible)
        assert len(violations) > 0

    def test_infeasible_score_is_zero(self, ft06, dummy_infeasible):
        validator = ConstraintValidator(ft06)
        assert validator.feasibility_score(dummy_infeasible) == 0.0

    def test_repaired_schedule_is_feasible(self, ft06, repaired_schedule):
        validator = ConstraintValidator(ft06)
        violations = validator.validate(repaired_schedule)
        assert len(violations) == 0

    def test_repaired_score_is_one(self, ft06, repaired_schedule):
        validator = ConstraintValidator(ft06)
        assert validator.feasibility_score(repaired_schedule) == 1.0

    def test_is_feasible_flag(self, ft06, dummy_infeasible, repaired_schedule):
        validator = ConstraintValidator(ft06)
        assert not validator.is_feasible(dummy_infeasible)
        assert validator.is_feasible(repaired_schedule)

    def test_machine_capacity_detected(self, ft06):
        """Two ops on same machine overlapping must be flagged."""
        op_a = ft06.jobs[0].operations[0]
        op_b = ft06.jobs[1].operations[0]
        # Force both onto machine 0, overlapping
        op_a_forced = Operation(op_a.job_id, op_a.step, 0, 10)
        op_b_forced = Operation(op_b.job_id, op_b.step, 0, 10)
        ops = [
            ScheduledOperation(op_a_forced, start_time=0,  end_time=10),
            ScheduledOperation(op_b_forced, start_time=5,  end_time=15),
        ]
        validator = ConstraintValidator(ft06)
        types = [v.constraint for v in validator.validate(ops)]
        assert "MachineCapacity" in types

    def test_precedence_violation_detected(self, ft06):
        """Step 1 starting before step 0 finishes must be flagged."""
        job = ft06.jobs[0]
        ops = [
            ScheduledOperation(job.operations[0], start_time=10, end_time=20),
            ScheduledOperation(job.operations[1], start_time=5,  end_time=15),
        ]
        validator = ConstraintValidator(ft06)
        types = [v.constraint for v in validator.validate(ops)]
        assert "JobPrecedence" in types


# ---------------------------------------------------------------------------
# Repair operator
# ---------------------------------------------------------------------------

class TestRepair:

    def test_repair_produces_feasible(self, ft06, dummy_infeasible):
        repaired = repair_schedule(dummy_infeasible, ft06)
        validator = ConstraintValidator(ft06)
        assert validator.is_feasible(repaired)

    def test_repair_preserves_all_operations(self, ft06, dummy_infeasible):
        repaired = repair_schedule(dummy_infeasible, ft06)
        assert len(repaired) == len(dummy_infeasible)

    def test_repair_nonnegative_starts(self, ft06, dummy_infeasible):
        repaired = repair_schedule(dummy_infeasible, ft06)
        assert all(s.start_time >= 0 for s in repaired)

    def test_repair_correct_end_times(self, ft06, dummy_infeasible):
        repaired = repair_schedule(dummy_infeasible, ft06)
        for sop in repaired:
            assert sop.end_time == sop.start_time + sop.operation.processing_time

    def test_repair_idempotent(self, ft06, dummy_infeasible):
        """Repairing an already-repaired schedule should change nothing."""
        repaired_once  = repair_schedule(dummy_infeasible, ft06)
        repaired_twice = repair_schedule(repaired_once, ft06)
        for a, b in zip(repaired_once, repaired_twice):
            assert a.start_time == b.start_time
            assert a.end_time   == b.end_time


# ---------------------------------------------------------------------------
# Schedule metrics
# ---------------------------------------------------------------------------

class TestScheduleMetrics:

    def test_metrics_computed(self, ft06, repaired_schedule):
        sched = Schedule(instance=ft06, scheduled_ops=repaired_schedule)
        metrics = sched.compute_metrics()
        assert set(metrics.keys()) == {"makespan", "tardiness", "utilization",
                                        "flowtime", "energy"}

    def test_makespan_positive(self, ft06, repaired_schedule):
        sched = Schedule(instance=ft06, scheduled_ops=repaired_schedule)
        assert sched.compute_metrics()["makespan"] > 0

    def test_utilization_in_range(self, ft06, repaired_schedule):
        sched = Schedule(instance=ft06, scheduled_ops=repaired_schedule)
        u = sched.compute_metrics()["utilization"]
        assert 0.0 <= u <= 1.0

    def test_fitness_positive(self, ft06, repaired_schedule):
        sched = Schedule(instance=ft06, scheduled_ops=repaired_schedule)
        assert sched.fitness() > 0

    def test_optimality_gap_positive(self, ft06, repaired_schedule):
        sched = Schedule(instance=ft06, scheduled_ops=repaired_schedule)
        gap = sched.optimality_gap()
        assert gap is not None
        assert gap >= 0.0

    def test_empty_schedule_returns_no_metrics(self, ft06):
        sched = Schedule(instance=ft06, scheduled_ops=[])
        assert sched.compute_metrics() == {}

    def test_empty_schedule_fitness_is_inf(self, ft06):
        sched = Schedule(instance=ft06, scheduled_ops=[])
        assert sched.fitness() == float("inf")