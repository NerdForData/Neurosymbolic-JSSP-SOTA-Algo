"""
Benchmark Plots
===============
Convergence curves and solver comparison charts.

Includes:
  - plot_convergence : GA fitness over generations (NS vs Pure GA)
  - plot_comparison  : bar chart comparing all solvers on all metrics
  - plot_metrics_radar: radar chart of multi-objective performance
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
import os


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------

def plot_convergence(
    ns_history:  List[float],
    ga_history:  List[float],
    instance_name: str = "",
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (10, 5),
) -> plt.Figure:
    """
    Plot GA convergence: NS Hybrid vs Pure GA side by side.

    This is the key visual proof that LLM seeding helps convergence.
    The NS Hybrid should start lower (better seeds) and converge faster.

    Args:
        ns_history:    best_per_gen from NS Hybrid GA run
        ga_history:    best_per_gen from Pure GA run
        instance_name: shown in title
        save_path:     file path to save
        show:          call plt.show()

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    gens = range(1, len(ns_history) + 1)

    ax.plot(gens, ns_history, color="#4C72B0", linewidth=2,
            label="NS Hybrid (LLM + GA)", zorder=3)
    ax.plot(range(1, len(ga_history) + 1), ga_history,
            color="#DD8452", linewidth=2, linestyle="--",
            label="Pure GA (random init)", zorder=3)

    # Shade improvement area
    min_len = min(len(ns_history), len(ga_history))
    ax.fill_between(
        range(1, min_len + 1),
        ns_history[:min_len],
        ga_history[:min_len],
        alpha=0.12, color="#4C72B0",
        label="LLM seeding advantage",
    )

    # Annotate final values
    ax.annotate(
        f"  {ns_history[-1]:.4f}",
        xy=(len(ns_history), ns_history[-1]),
        fontsize=9, color="#4C72B0", fontweight="bold",
    )
    ax.annotate(
        f"  {ga_history[-1]:.4f}",
        xy=(len(ga_history), ga_history[-1]),
        fontsize=9, color="#DD8452", fontweight="bold",
    )

    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Best Fitness (lower = better)", fontsize=11)
    title = "Convergence: NS Hybrid vs Pure GA"
    if instance_name:
        title += f"  |  Instance: {instance_name.upper()}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_xlim(1, max(len(ns_history), len(ga_history)))

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Solver comparison bar chart
# ---------------------------------------------------------------------------

def plot_comparison(
    df: pd.DataFrame,
    metric: str = "makespan",
    instance: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Bar chart comparing all solvers on a given metric.

    Args:
        df:       Results DataFrame from BenchmarkRunner
        metric:   Column to plot (makespan, fitness, tardiness, etc.)
        instance: Filter to one instance (None = average across all)
        save_path: file path to save
        show:     call plt.show()

    Returns:
        matplotlib Figure
    """
    if instance:
        data = df[df["instance"] == instance]
        subtitle = f"Instance: {instance.upper()}"
    else:
        data = df
        subtitle = "Averaged across all instances"

    # Mean ± std per solver
    grouped = data.groupby("solver")[metric].agg(["mean", "std"]).reset_index()
    grouped = grouped.sort_values("mean")

    solvers = grouped["solver"].tolist()
    means   = grouped["mean"].tolist()
    stds    = grouped["std"].fillna(0).tolist()

    # Color NS Hybrid differently
    colors = []
    for s in solvers:
        if "NS" in s or "Hybrid" in s:
            colors.append("#4C72B0")
        elif "PureGA" in s:
            colors.append("#DD8452")
        elif "PSO" in s:
            colors.append("#55A868")
        elif "OR-Tools" in s:
            colors.append("#8172B3")
        else:
            colors.append("#8C8C8C")

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(solvers))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white",
                  linewidth=0.8, alpha=0.88, zorder=3)

    # Value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + max(means) * 0.01,
            f"{mean:.1f}",
            ha="center", va="bottom", fontsize=8, fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(solvers, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel(metric.replace("_", " ").title(), fontsize=11)
    ax.set_title(
        f"Solver Comparison — {metric.replace('_', ' ').title()}\n{subtitle}",
        fontsize=12, fontweight="bold",
    )
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # For utilization: higher is better
    if metric == "utilization":
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        ax.set_ylabel("Machine Utilization", fontsize=11)

    # Legend
    legend_items = [
        mpatches.Patch(color="#4C72B0", label="NS Hybrid"),
        mpatches.Patch(color="#DD8452", label="Pure GA"),
        mpatches.Patch(color="#55A868", label="PSO"),
        mpatches.Patch(color="#8172B3", label="OR-Tools"),
        mpatches.Patch(color="#8C8C8C", label="Dispatching Rules"),
    ]
    ax.legend(handles=legend_items, fontsize=8, loc="upper left",
              framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Multi-metric radar chart
# ---------------------------------------------------------------------------

def plot_radar(
    df: pd.DataFrame,
    solvers: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (8, 8),
) -> plt.Figure:
    """
    Radar (spider) chart showing multi-objective performance per solver.
    Each axis is normalized so all metrics are comparable.
    Lower = better for all axes (utilization is inverted).

    Args:
        df:       Results DataFrame
        solvers:  Subset of solvers to show (None = all)
        save_path: file path to save
        show:     call plt.show()

    Returns:
        matplotlib Figure
    """
    import matplotlib.patches as mpatches

    metrics = ["makespan", "tardiness", "flowtime", "energy", "utilization"]
    labels  = ["Makespan", "Tardiness", "Flow Time", "Energy", "Utilization\n(inverted)"]

    if solvers is None:
        solvers = df["solver"].unique().tolist()

    # Compute mean per solver
    means = df.groupby("solver")[metrics].mean()

    # Normalize each metric to [0,1]  (0=best, 1=worst)
    norm = means.copy()
    for m in metrics:
        col = means[m]
        mn, mx = col.min(), col.max()
        if mx > mn:
            if m == "utilization":
                norm[m] = 1 - (col - mn) / (mx - mn)  # invert
            else:
                norm[m] = (col - mn) / (mx - mn)
        else:
            norm[m] = 0.0

    # Radar setup
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"polar": True})

    colors_map = {
        s: ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",
            "#937860","#DA8BC3","#8C8C8C"][i % 8]
        for i, s in enumerate(solvers)
    }

    for solver in solvers:
        if solver not in norm.index:
            continue
        values = norm.loc[solver, metrics].tolist()
        values += values[:1]
        color = colors_map[solver]
        ax.plot(angles, values, color=color, linewidth=2, label=solver)
        ax.fill(angles, values, color=color, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=7)
    ax.set_title("Multi-Objective Performance\n(lower = better on all axes)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Optimality gap comparison
# ---------------------------------------------------------------------------

def plot_optimality_gaps(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
    show: bool = False,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Line chart of optimality gap per solver per instance.
    Only plotted for instances that have a known BKS.
    """
    data = df[df["optimality_gap"].notna()].copy()
    if data.empty:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No BKS data available",
                ha="center", va="center", transform=ax.transAxes, fontsize=14)
        return fig

    grouped = data.groupby(["instance", "solver"])["optimality_gap"].mean().reset_index()
    instances = sorted(grouped["instance"].unique())
    solvers   = grouped["solver"].unique()

    fig, ax = plt.subplots(figsize=figsize)
    markers = ["o", "s", "^", "D", "v", "P", "X"]

    for i, solver in enumerate(solvers):
        sub = grouped[grouped["solver"] == solver]
        sub = sub.set_index("instance").reindex(instances)
        color = (["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3",
                  "#937860","#8C8C8C"][i % 7])
        ax.plot(instances, sub["optimality_gap"].values,
                marker=markers[i % len(markers)],
                color=color, linewidth=2, markersize=7,
                label=solver)

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1,
               label="Optimal (gap=0%)")
    ax.set_xlabel("Instance", fontsize=11)
    ax.set_ylabel("Optimality Gap (%)", fontsize=11)
    ax.set_title("Optimality Gap vs Best Known Solution",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig