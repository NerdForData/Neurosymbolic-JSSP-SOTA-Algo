"""
Sequence Decoder
================
Converts a priority sequence (list of job IDs) into a fully timed,
feasible Schedule using the Giffler-Thompson active schedule algorithm.

This is the bridge between the Neuro (LLM) and Symbolic (GA) components:
  LLM outputs a sequence → decoder produces a Schedule → GA evaluates fitness

The decoder guarantees feasibility by construction (no repair needed).
"""

from typing import List
from src.problem.jssp import (
    JSSPInstance, Schedule, ScheduledOperation
)


def decode_sequence(
    sequence: List[int],
    instance: JSSPInstance,
) -> Schedule:
    """
    Decode a priority sequence into a feasible Schedule.

    Uses the Giffler-Thompson active schedule generation algorithm:
      - Process job IDs in the order given by the sequence
      - For each job ID, schedule its NEXT unscheduled operation
      - Start time = max(machine_available, job_available)
      - This guarantees both machine capacity and precedence constraints

    Args:
        sequence: List of job IDs, length = n_jobs * n_machines
                  Each job ID appears exactly n_machines times
        instance: The JSSP instance

    Returns:
        A fully feasible Schedule with all metrics computable
    """
    n_jobs     = instance.num_jobs
    n_machines = instance.num_machines

    # Track next operation step for each job
    next_step = [0] * n_jobs

    # Track earliest available time per machine and per job
    machine_available = [0] * n_machines
    job_available     = [0] * n_jobs

    scheduled_ops = []

    for job_id in sequence:
        step = next_step[job_id]

        # Safety: skip if this job is already fully scheduled
        if step >= instance.jobs[job_id].num_operations:
            continue

        op = instance.jobs[job_id].operations[step]
        m  = op.machine_id
        p  = op.processing_time

        # Earliest feasible start
        start = max(machine_available[m], job_available[job_id])
        end   = start + p

        machine_available[m]   = end
        job_available[job_id]  = end
        next_step[job_id]     += 1

        scheduled_ops.append(ScheduledOperation(
            operation  = op,
            start_time = start,
            end_time   = end,
        ))

    return Schedule(instance=instance, scheduled_ops=scheduled_ops)


def decode_population(
    sequences: List[List[int]],
    instance: JSSPInstance,
) -> List[Schedule]:
    """
    Decode a full population of sequences into Schedules.

    Args:
        sequences: List of priority sequences
        instance:  The JSSP instance

    Returns:
        List of Schedule objects, one per sequence
    """
    return [decode_sequence(seq, instance) for seq in sequences]