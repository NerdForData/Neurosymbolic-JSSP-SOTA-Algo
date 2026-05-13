"""
JSSP Problem Definition
=======================
Formal data structures for the multi-objective Job Shop Scheduling Problem.
Covers all constraints: machine capacity, job precedence, due dates, makespan,
tardiness, machine utilization, flow time, and energy consumption.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Operation:
    """A single operation: one job step processed on one machine."""
    job_id: int
    step: int
    machine_id: int
    processing_time: int
    energy_rate: float = 1.0


@dataclass
class Job:
    """A job is an ordered sequence of operations."""
    job_id: int
    operations: List[Operation]
    due_date: Optional[int] = None
    release_time: int = 0
    weight: float = 1.0

    @property
    def num_operations(self) -> int:
        return len(self.operations)

    @property
    def total_processing_time(self) -> int:
        return sum(op.processing_time for op in self.operations)


@dataclass
class Machine:
    """A machine that processes operations one at a time."""
    machine_id: int
    name: str = ""
    idle_energy_rate: float = 0.1
    active_energy_rate: float = 1.0


@dataclass
class JSSPInstance:
    """
    Complete JSSP instance.
    Holds all jobs, machines, and objective weights for multi-objective fitness.
    """
    name: str
    jobs: List[Job]
    machines: List[Machine]
    best_known_solution: Optional[int] = None

    # Multi-objective weights (must sum to 1.0)
    w_makespan: float = 0.35
    w_tardiness: float = 0.25
    w_utilization: float = 0.15
    w_flowtime: float = 0.15
    w_energy: float = 0.10

    @property
    def num_jobs(self) -> int:
        return len(self.jobs)

    @property
    def num_machines(self) -> int:
        return len(self.machines)

    @property
    def num_operations(self) -> int:
        return sum(j.num_operations for j in self.jobs)

    def __repr__(self):
        bks = f", BKS={self.best_known_solution}" if self.best_known_solution else ""
        return f"JSSPInstance('{self.name}', {self.num_jobs}j x {self.num_machines}m{bks})"


@dataclass
class ScheduledOperation:
    """An operation placed on the timeline with a concrete start and end time."""
    operation: Operation
    start_time: int
    end_time: int

    @property
    def job_id(self):
        return self.operation.job_id

    @property
    def machine_id(self):
        return self.operation.machine_id

    @property
    def processing_time(self):
        return self.operation.processing_time


@dataclass
class Schedule:
    """
    A complete schedule: every operation assigned to a start time.
    Computes all multi-objective metrics on demand.
    """
    instance: JSSPInstance
    scheduled_ops: List[ScheduledOperation] = field(default_factory=list)
    _metrics: Dict = field(default_factory=dict, repr=False)

    def compute_metrics(self) -> Dict[str, float]:
        """Compute makespan, tardiness, utilization, flowtime, energy."""
        if not self.scheduled_ops:
            return {}

        n_jobs = self.instance.num_jobs
        n_machines = self.instance.num_machines

        # Completion time per job
        completion = {j: 0 for j in range(n_jobs)}
        for sop in self.scheduled_ops:
            completion[sop.job_id] = max(completion[sop.job_id], sop.end_time)

        makespan = max(completion.values())

        # Weighted tardiness
        total_tardiness = 0.0
        for job in self.instance.jobs:
            if job.due_date is not None:
                late = max(0, completion[job.job_id] - job.due_date)
                total_tardiness += job.weight * late

        # Total flow time
        total_flowtime = sum(
            completion[job.job_id] - job.release_time
            for job in self.instance.jobs
        )

        # Machine utilization
        machine_busy = {m: 0 for m in range(n_machines)}
        for sop in self.scheduled_ops:
            machine_busy[sop.machine_id] += sop.processing_time
        utilization = (
            sum(machine_busy.values()) / (n_machines * makespan)
            if makespan > 0 else 0.0
        )

        # Energy (active + idle)
        energy = 0.0
        for sop in self.scheduled_ops:
            energy += sop.operation.energy_rate * sop.processing_time
        for m_id, machine in enumerate(self.instance.machines):
            idle_time = makespan - machine_busy[m_id]
            energy += machine.idle_energy_rate * idle_time

        self._metrics = {
            "makespan": makespan,
            "tardiness": total_tardiness,
            "utilization": utilization,
            "flowtime": total_flowtime,
            "energy": energy,
        }
        return self._metrics

    def fitness(self) -> float:
        """
        Weighted multi-objective fitness (lower = better).
        Each objective is normalized before weighting.
        """
        m = self.compute_metrics()
        if not m:
            return float("inf")

        inst = self.instance
        total_proc = max(sum(j.total_processing_time for j in inst.jobs), 1)

        norm_makespan    = m["makespan"]  / total_proc
        norm_tardiness   = m["tardiness"] / total_proc
        norm_utilization = 1.0 - m["utilization"]   # invert: higher = better
        norm_flowtime    = m["flowtime"]  / max(total_proc * inst.num_jobs, 1)
        norm_energy      = m["energy"]    / max(total_proc * 2, 1)

        return (
            inst.w_makespan    * norm_makespan    +
            inst.w_tardiness   * norm_tardiness   +
            inst.w_utilization * norm_utilization +
            inst.w_flowtime    * norm_flowtime    +
            inst.w_energy      * norm_energy
        )

    def optimality_gap(self) -> Optional[float]:
        """Percentage gap above best known solution (None if BKS unavailable)."""
        bks = self.instance.best_known_solution
        m = self._metrics or self.compute_metrics()
        if bks and m:
            return 100.0 * (m["makespan"] - bks) / bks
        return None