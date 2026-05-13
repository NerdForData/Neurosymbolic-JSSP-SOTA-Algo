"""
LLM Prompt Builder
==================
Converts a JSSPInstance into a structured natural-language prompt
that guides the LLM to generate good job priority sequences.

Design philosophy:
  - The LLM does NOT produce a full Gantt chart (too complex, error-prone)
  - It produces a PRIORITY SEQUENCE: an ordering of job IDs
  - This sequence is decoded by the GA into a feasible schedule
  - Few-shot examples teach the LLM the output format
"""

from typing import List
from src.problem.jssp import JSSPInstance


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert manufacturing scheduling assistant specializing 
in Job Shop Scheduling Problems (JSSP). Your task is to analyze scheduling instances 
and suggest intelligent job priority sequences that minimize makespan, tardiness, 
and energy consumption while maximizing machine utilization.

A priority sequence is a list of job IDs where each job ID appears exactly 
n_machines times (once per operation). The sequence determines the order in which 
operations are dispatched to machines.

You must respond ONLY with a JSON object in exactly this format:
{
  "sequences": [
    [job_id, job_id, ...],
    [job_id, job_id, ...],
    ...
  ],
  "reasoning": "brief explanation of your strategy"
}

Rules:
- Each sequence must have exactly n_jobs * n_machines integers
- Each job ID (0 to n_jobs-1) must appear exactly n_machines times per sequence
- Use scheduling heuristics: SPT (shortest processing time first), 
  LPT (longest processing time first), EDD (earliest due date first),
  or combinations based on the instance characteristics
- Generate diverse sequences using different heuristic strategies
"""


FEW_SHOT_EXAMPLE = """
Example for a 3-job x 3-machine instance:
Jobs:
  Job 0: M0(p=3), M1(p=2), M2(p=4)  due=15
  Job 1: M1(p=5), M0(p=1), M2(p=3)  due=12
  Job 2: M2(p=2), M1(p=4), M0(p=6)  due=18

Response:
{
  "sequences": [
    [1, 2, 0, 1, 2, 0, 0, 1, 2],
    [0, 1, 2, 0, 1, 2, 0, 1, 2],
    [2, 0, 1, 2, 0, 1, 2, 0, 1]
  ],
  "reasoning": "Sequence 1 prioritizes Job 1 (tightest due date, EDD), 
  Sequence 2 uses round-robin for diversity, Sequence 3 uses LPT to 
  keep machines busy."
}
"""


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(instance: JSSPInstance, n_sequences: int = 5) -> str:
    """
    Build a complete user prompt describing the JSSP instance.

    Args:
        instance:    The JSSP instance to schedule
        n_sequences: How many diverse sequences to request from the LLM

    Returns:
        A formatted prompt string ready to send to the LLM
    """
    lines = []

    # Header
    lines.append(f"Generate {n_sequences} diverse priority sequences for this "
                 f"Job Shop Scheduling instance.\n")

    # Instance summary
    lines.append(f"Instance: {instance.name}")
    lines.append(f"Jobs: {instance.num_jobs}, Machines: {instance.num_machines}")
    lines.append(f"Total operations per sequence: "
                 f"{instance.num_jobs * instance.num_machines}\n")

    # Objectives
    lines.append("Optimization objectives (weights):")
    lines.append(f"  - Makespan minimization     : {instance.w_makespan:.0%}")
    lines.append(f"  - Tardiness minimization    : {instance.w_tardiness:.0%}")
    lines.append(f"  - Machine utilization       : {instance.w_utilization:.0%}")
    lines.append(f"  - Flow time minimization    : {instance.w_flowtime:.0%}")
    lines.append(f"  - Energy consumption        : {instance.w_energy:.0%}\n")

    # Job details
    lines.append("Job details (format: Step->Machine(processing_time)):")
    for job in instance.jobs:
        ops_str = "  ->  ".join(
            f"M{op.machine_id}(p={op.processing_time})"
            for op in job.operations
        )
        due_str = f"  [due={job.due_date}]" if job.due_date else ""
        total = job.total_processing_time
        lines.append(f"  Job {job.job_id} (total={total}){due_str}: {ops_str}")

    lines.append("")

    # Machine load summary (helps LLM reason about bottlenecks)
    machine_load = {m: 0 for m in range(instance.num_machines)}
    for job in instance.jobs:
        for op in job.operations:
            machine_load[op.machine_id] += op.processing_time

    lines.append("Machine load (total processing time assigned):")
    for m_id, load in sorted(machine_load.items(), key=lambda x: -x[1]):
        bar = "█" * (load // 10)
        lines.append(f"  Machine {m_id}: {load:4d}  {bar}")
    lines.append("")

    # Bottleneck hint
    bottleneck = max(machine_load, key=machine_load.get)
    lines.append(f"Bottleneck machine: M{bottleneck} "
                 f"(load={machine_load[bottleneck]}). "
                 f"Prioritize jobs that use M{bottleneck} early.\n")

    # SPT hint
    jobs_by_total = sorted(instance.jobs, key=lambda j: j.total_processing_time)
    spt_order = [j.job_id for j in jobs_by_total]
    lines.append(f"SPT job order (shortest total first): {spt_order}")

    # EDD hint
    if all(j.due_date for j in instance.jobs):
        jobs_by_due = sorted(instance.jobs, key=lambda j: j.due_date)
        edd_order = [j.job_id for j in jobs_by_due]
        lines.append(f"EDD job order (earliest due date first): {edd_order}")

    lines.append("")
    lines.append(f"Generate exactly {n_sequences} sequences. "
                 f"Each sequence must contain {instance.num_jobs * instance.num_machines} "
                 f"integers. Each job ID (0 to {instance.num_jobs - 1}) must appear "
                 f"exactly {instance.num_machines} times per sequence.")

    return "\n".join(lines)


def build_messages(instance: JSSPInstance, n_sequences: int = 5) -> List[dict]:
    """
    Build the full messages list for the chat completion API.

    Returns:
        List of message dicts ready for groq/openai client
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT + "\n" + FEW_SHOT_EXAMPLE},
        {"role": "user",   "content": build_prompt(instance, n_sequences)},
    ]