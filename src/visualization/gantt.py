"""
Gantt Chart Visualizer
======================
Produces publication-quality Gantt charts from a Schedule object.

Features:
  - One row per machine, colored by job
  - Tardiness highlighted in red outline
  - Makespan line marked
  - Due date indicators
  - Saves to PNG/PDF for the client report
"""

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (works without display)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from typing import Optional
import os

from src.problem.jssp import Schedule, JSSPInstance


# Fixed color palette — consistent across all charts
JOB_COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
]


def plot_gantt(
    schedule: Schedule,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot a Gantt chart for a given schedule.

    Args:
        schedule:  The Schedule to visualize
        title:     Chart title (auto-generated if None)
        save_path: File path to save (PNG or PDF). None = don't save.
        show:      Whether to call plt.show() (use False in scripts)
        figsize:   Figure size in inches

    Returns:
        matplotlib Figure object
    """
    instance = schedule.instance
    metrics  = schedule.compute_metrics()
    n_machines = instance.num_machines
    n_jobs     = instance.num_jobs
    makespan   = metrics["makespan"]

    fig, ax = plt.subplots(figsize=figsize)

    # Compute completion times per job for tardiness check
    completion = {}
    for sop in schedule.scheduled_ops:
        completion[sop.job_id] = max(
            completion.get(sop.job_id, 0), sop.end_time
        )

    # Draw operations
    for sop in schedule.scheduled_ops:
        job_id    = sop.job_id
        machine   = sop.machine_id
        start     = sop.start_time
        duration  = sop.processing_time
        color     = JOB_COLORS[job_id % len(JOB_COLORS)]

        # Check if this job is tardy
        due_date  = instance.jobs[job_id].due_date
        is_tardy  = due_date and completion[job_id] > due_date

        # Draw the operation block
        bar = FancyBboxPatch(
            (start, machine - 0.4),
            duration, 0.8,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor="#CC0000" if is_tardy else "white",
            linewidth=2.0 if is_tardy else 0.5,
            alpha=0.85,
        )
        ax.add_patch(bar)

        # Label if wide enough
        if duration > makespan * 0.03:
            ax.text(
                start + duration / 2, machine,
                f"J{job_id}",
                ha="center", va="center",
                fontsize=7, fontweight="bold",
                color="white",
            )

    # Makespan vertical line
    ax.axvline(x=makespan, color="#CC0000", linestyle="--",
               linewidth=1.5, label=f"Makespan={makespan:.0f}", zorder=5)

    # Due date vertical line (if uniform)
    due_dates = [j.due_date for j in instance.jobs if j.due_date]
    if due_dates and len(set(due_dates)) == 1:
        ax.axvline(x=due_dates[0], color="orange", linestyle=":",
                   linewidth=1.5, label=f"Due date={due_dates[0]:.0f}", zorder=4)

    # Axes
    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f"M{m}" for m in range(n_machines)], fontsize=9)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Machine", fontsize=11)
    ax.set_xlim(0, makespan * 1.05)
    ax.set_ylim(-0.6, n_machines - 0.4)
    ax.invert_yaxis()

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle="--", alpha=0.4)

    # Title
    if title is None:
        gap_str = ""
        gap = schedule.optimality_gap()
        if gap is not None:
            gap_str = f"  |  Gap={gap:.1f}%"
        title = (f"Schedule: {instance.name.upper()}  "
                 f"|  Makespan={makespan:.0f}{gap_str}  "
                 f"|  Utilization={metrics['utilization']:.1%}")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

    # Job legend
    handles = [
        mpatches.Patch(
            color=JOB_COLORS[j % len(JOB_COLORS)],
            label=f"Job {j}"
        )
        for j in range(n_jobs)
    ]
    # Add makespan/due date to legend
    handles.append(plt.Line2D([0], [0], color="#CC0000", linestyle="--",
                               label=f"Makespan={makespan:.0f}"))
    if due_dates and len(set(due_dates)) == 1:
        handles.append(plt.Line2D([0], [0], color="orange", linestyle=":",
                                   label=f"Due={due_dates[0]:.0f}"))

    ax.legend(
        handles=handles,
        loc="upper right",
        fontsize=7,
        ncol=max(1, n_jobs // 6 + 1),
        framealpha=0.9,
    )

    # Tardiness annotation
    tardiness = metrics["tardiness"]
    if tardiness > 0:
        ax.text(
            0.01, 0.02,
            f"Total tardiness: {tardiness:.0f}  (red border = tardy job)",
            transform=ax.transAxes,
            fontsize=8, color="#CC0000",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig