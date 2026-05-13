"""
Constraint Validator & Schedule Repair
=======================================
The symbolic guarantee layer of the neuro-symbolic system.

Every schedule produced by any module (LLM seeds, GA operators, dispatching
rules) must pass through validate() before being accepted.  If it fails,
repair_schedule() fixes it by construction -- guaranteeing feasibility.

Hard constraints checked:
  1. NonNegativeStart   -- all operations start at time >= 0
  2. ProcessingTime     -- end_time == start_time + processing_time
  3. MachineCapacity    -- no two operations overlap on the same machine
  4. JobPrecedence      -- each step starts only after the previous step finishes
"""

from collections import defaultdict
from typing import List
from src.problem.jssp import JSSPInstance, ScheduledOperation


# ---------------------------------------------------------------------------
# Violation descriptor
# ---------------------------------------------------------------------------

class ConstraintViolation:
    def __init__(self, constraint: str, description: str, severity: str = "hard"):
        self.constraint  = constraint
        self.description = description
        self.severity    = severity

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.constraint}: {self.description}"


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class ConstraintValidator:
    """
    Validates all hard constraints on a list of ScheduledOperations.
    An empty violation list means the schedule is fully feasible.
    """

    def __init__(self, instance: JSSPInstance):
        self.instance = instance

    def validate(self, ops: List[ScheduledOperation]) -> List[ConstraintViolation]:
        """Return all constraint violations (empty = feasible)."""
        v = []
        v.extend(self._check_nonnegative_start(ops))
        v.extend(self._check_processing_times(ops))
        v.extend(self._check_machine_capacity(ops))
        v.extend(self._check_job_precedence(ops))
        return v

    def is_feasible(self, ops: List[ScheduledOperation]) -> bool:
        """True only when there are zero hard violations."""
        return len(self.validate(ops)) == 0

    def feasibility_score(self, ops: List[ScheduledOperation]) -> float:
        """
        Score in [0, 1].  1.0 = fully feasible.
        Used by the GA fitness function to penalise infeasible individuals
        without immediately discarding them.
        """
        violations = [v for v in self.validate(ops) if v.severity == "hard"]
        if not violations:
            return 1.0
        return max(0.0, 1.0 - len(violations) / max(len(ops), 1))

    # ── private checkers ────────────────────────────────────────────────

    def _check_nonnegative_start(self, ops):
        return [
            ConstraintViolation(
                "NonNegativeStart",
                f"Op(job={s.job_id}, step={s.operation.step}) start={s.start_time} < 0"
            )
            for s in ops if s.start_time < 0
        ]

    def _check_processing_times(self, ops):
        violations = []
        for s in ops:
            expected = s.start_time + s.operation.processing_time
            if s.end_time != expected:
                violations.append(ConstraintViolation(
                    "ProcessingTime",
                    f"Op(job={s.job_id}, step={s.operation.step}) "
                    f"end={s.end_time} != start+p={expected}"
                ))
        return violations

    def _check_machine_capacity(self, ops):
        """No two operations may overlap on the same machine."""
        violations = []
        by_machine = defaultdict(list)
        for s in ops:
            by_machine[s.machine_id].append(s)

        for m_id, m_ops in by_machine.items():
            sorted_ops = sorted(m_ops, key=lambda x: x.start_time)
            for i in range(len(sorted_ops) - 1):
                a, b = sorted_ops[i], sorted_ops[i + 1]
                if a.end_time > b.start_time:
                    violations.append(ConstraintViolation(
                        "MachineCapacity",
                        f"Machine {m_id}: job={a.job_id} [{a.start_time},{a.end_time}) "
                        f"overlaps job={b.job_id} [{b.start_time},{b.end_time})"
                    ))
        return violations

    def _check_job_precedence(self, ops):
        """Each operation must start only after the previous step of the same job ends."""
        violations = []
        by_job = defaultdict(list)
        for s in ops:
            by_job[s.job_id].append(s)

        for j_id, j_ops in by_job.items():
            sorted_ops = sorted(j_ops, key=lambda x: x.operation.step)
            for i in range(len(sorted_ops) - 1):
                prev, curr = sorted_ops[i], sorted_ops[i + 1]
                if curr.start_time < prev.end_time:
                    violations.append(ConstraintViolation(
                        "JobPrecedence",
                        f"Job {j_id}: step {curr.operation.step} starts at "
                        f"{curr.start_time} before step {prev.operation.step} "
                        f"ends at {prev.end_time}"
                    ))
        return violations


# ---------------------------------------------------------------------------
# Repair operator
# ---------------------------------------------------------------------------

def repair_schedule(
    scheduled_ops: List[ScheduledOperation],
    instance: JSSPInstance,
) -> List[ScheduledOperation]:
    """
    Repair any infeasible schedule to make it fully feasible.

    This is the symbolic repair operator used after LLM-generated seeds arrive
    in potentially infeasible form, and after GA crossover/mutation.

    Strategy (earliest-feasible-start):
      - Process operations in job-step order (respects precedence by construction)
      - For each operation, start = max(machine_free, job_free)
      - This guarantees no machine overlap and no precedence violation

    Returns a new list of ScheduledOperations; the input is not modified.
    """
    n_machines = instance.num_machines
    n_jobs     = instance.num_jobs

    machine_free = [0] * n_machines   # earliest time machine m is available
    job_free     = [0] * n_jobs       # earliest time job j can start its next op

    # Sort by step first so earlier steps are scheduled before later ones
    sorted_ops = sorted(scheduled_ops, key=lambda s: (s.operation.step, s.job_id))

    repaired = []
    for sop in sorted_ops:
        m = sop.machine_id
        j = sop.job_id
        p = sop.operation.processing_time

        start = max(machine_free[m], job_free[j])
        end   = start + p

        machine_free[m] = end
        job_free[j]     = end

        repaired.append(ScheduledOperation(
            operation=sop.operation,
            start_time=start,
            end_time=end,
        ))

    return repaired