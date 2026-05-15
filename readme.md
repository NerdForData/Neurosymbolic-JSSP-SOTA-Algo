# Neuro-Symbolic Hybrid Solver for Job Shop Scheduling

A production-ready AI system for manufacturing scheduling that combines a **Large Language Model (LLM)** with a **constraint-aware Genetic Algorithm (GA)** to optimise multi-objective Job Shop Scheduling Problems.

> **Language + Guarantees + Generalisation** — the three pillars of neuro-symbolic AI applied to manufacturing.

---

## What It Does

Given a set of jobs, machines, and constraints, the system produces an optimal production schedule that simultaneously minimises:

- Makespan (total production time)
- Tardiness (late delivery penalties)
- Flow time (work-in-progress time)
- Energy consumption
- Machine idle time (maximises utilisation)

Every schedule produced is **guaranteed feasible** — hard constraints (machine capacity, job precedence) are never violated.

---

## How It Works

```
┌─────────────────────────────────────────────────┐
│             NS Hybrid Solver                     │
│                                                  │
│  ┌─────────────┐      ┌──────────────────────┐  │
│  │ LLM Module  │ ───▶ │     GA Engine        │  │
│  │  (Neuro)    │      │    (Symbolic)        │  │
│  │             │      │                      │  │
│  │ Groq API    │      │ Constraint-aware     │  │
│  │ Llama 3.1   │      │ OX Crossover         │  │
│  │             │      │ Swap/Insert Mutation │  │
│  │ Generates   │      │ Tournament Selection │  │
│  │ smart seeds │      │ Elitism + Diversity  │  │
│  └─────────────┘      └──────────────────────┘  │
│          │                       │               │
│          ▼                       ▼               │
│  ┌───────────────────────────────────────────┐  │
│  │       Symbolic Guarantee Layer            │  │
│  │  Constraint Validator + Schedule Repair   │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

1. **LLM seeding** — Groq (Llama 3.1 8B) analyses the instance and generates diverse priority sequences using SPT, EDD, and LPT heuristic reasoning
2. **GA evolution** — a constraint-aware GA evolves the population over 200 generations; elitism and diversity injection prevent premature convergence
3. **Stagnation recovery** — if the GA stagnates, the LLM is queried again at higher temperature for fresh diversity
4. **Symbolic guarantee** — every schedule is validated against hard constraints; infeasible outputs are repaired automatically

---

## Project Structure

```
nsai_jss/
│
├── .env                          # GROQ_API_KEY=your_key_here
├── conftest.py                   # pytest path configuration
├── requirements.txt
├── generate_plots.py             # generates all charts → results/plots/
├── build_report.js               # generates client_report.docx
│
├── src/
│   ├── ns_solver.py              # NS integration loop (main entry point)
│   │
│   ├── problem/                  # Phase 1 — problem definition
│   │   ├── jssp.py               # core data structures + metrics
│   │   ├── loader.py             # benchmark instance loader
│   │   └── constraints.py        # constraint validator + repair
│   │
│   ├── llm/                      # Phase 2 — neuro component
│   │   ├── prompt_builder.py     # structured prompts from instance data
│   │   ├── llm_client.py         # Groq API wrapper + heuristic fallback
│   │   └── decoder.py            # priority sequence → feasible schedule
│   │
│   ├── ga/                       # Phase 2 — symbolic component
│   │   └── genetic_algorithm.py  # OX crossover, mutation, elitism
│   │
│   ├── benchmark/                # Phase 3 — benchmarking
│   │   ├── baseline.py           # SPT/LPT/EDD, Pure GA, PSO, OR-Tools
│   │   ├── runner.py             # unified benchmark harness
│   │   └── analysis.py           # Wilcoxon tests, ablation summary
│   │
│   └── visualization/            # Phase 4 — charts
│       ├── gantt.py              # Gantt chart generator
│       └── plots.py              # convergence, comparison, radar charts
│
├── instances/                    # downloaded benchmark files (auto-populated)
├── results/                      # benchmark CSVs + charts saved here
│   └── plots/
│
├── notebooks/
│   └── ns_hybrid_demo.ipynb      # end-to-end demo notebook
│
└── tests/
    ├── test_phase1.py            # 27 tests — problem definition
    ├── test_phase2.py            # 34 tests — LLM + GA
    ├── test_phase3.py            # 30 tests — benchmarking
    └── test_phase4.py            # 21 tests — visualization
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <your-repo-url>
cd nsai_jss
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up Groq API key

Sign up for a free key at [console.groq.com](https://console.groq.com), then:

```bash
# Create .env file in nsai_jss/ root
echo "GROQ_API_KEY=your_key_here" > .env
```

> The system works without a key — it falls back to heuristic seeding automatically. The LLM key unlocks better initial populations and faster convergence.

### 3. Run tests

```bash
pytest tests/ -v
```

Expected: **112 tests passing** across all 4 phases.

### 4. Solve a scheduling instance

```python
from src.problem.loader import load_instance
from src.ns_solver import NSHybridSolver
from src.ga.genetic_algorithm import GAConfig

instance = load_instance("ft06")   # built-in, no download needed

solver = NSHybridSolver(
    ga_config   = GAConfig(population_size=50, n_generations=200),
    n_llm_seeds = 5,
)

result = solver.solve(instance, verbose=True)
print(result.summary())
```

### 5. Generate charts

```bash
python generate_plots.py
```

Charts are saved to `results/plots/`. Takes ~2 minutes.

### 6. Run the full benchmark

```python
from src.benchmark.runner import BenchmarkRunner, BenchmarkConfig

runner = BenchmarkRunner(BenchmarkConfig(
    instances      = ["ft06", "ft10", "ta01", "ta02", "ta03"],
    n_runs         = 5,
    ga_generations = 200,
))
runner.run()
runner.save("results/benchmark_results.csv")
```

### 7. View results in notebook

```bash
cd notebooks
jupyter notebook ns_hybrid_demo.ipynb
```

### 8. Generate client report

```bash
# Requires Node.js
npm install -g docx
node build_report.js
# Output: client_report.docx
```

---

## Benchmark Instances

| Tier | Instances | Size | Notes |
|---|---|---|---|
| 1 — Small | ft06, ft10, ft20 | 6–20 jobs | Built-in, no download |
| 2 — Medium | ta01–ta30 | 15×15 to 20×20 | Auto-downloaded on first use |
| 3 — Large | ta31–ta80 | 30×20 to 100×20 | Auto-downloaded on first use |

To pre-download all Tier 2 instances:

```python
from src.problem.loader import download_all_tier
download_all_tier(tier=2)   # saves to instances/
```

---

## Competitor Algorithms

The system benchmarks against five baselines:

| Algorithm | Type | Purpose |
|---|---|---|
| Dispatching rules (SPT, LPT, EDD, MWR) | Industrial baseline | What most factories use today |
| Pure GA | Metaheuristic | Ablation — isolates LLM contribution |
| PSO | Metaheuristic | Standard SOTA comparison |
| OR-Tools CP-SAT | Exact solver | Quality upper bound |
| **NS Hybrid (ours)** | Neuro-Symbolic | Proposed system |

The ablation study (Pure GA vs NS Hybrid) directly quantifies the value of LLM seeding.

---

## Configuration

All GA and solver parameters are configurable via `GAConfig` and `BenchmarkConfig`:

```python
from src.ga.genetic_algorithm import GAConfig

cfg = GAConfig(
    population_size  = 50,     # individuals per generation
    n_generations    = 200,    # evolution steps
    crossover_rate   = 0.85,   # OX crossover probability
    mutation_rate    = 0.15,   # swap mutation probability
    tournament_size  = 3,      # selection pressure
    elitism_count    = 2,      # top-k always survive
    diversity_inject = 10,     # re-inject randoms every N gens
    seed             = 42,     # reproducibility
)
```

Objective weights are set per instance in `JSSPInstance`:

```python
instance.w_makespan    = 0.35
instance.w_tardiness   = 0.25
instance.w_utilization = 0.15
instance.w_flowtime    = 0.15
instance.w_energy      = 0.10
```

---

## LLM Options

| Provider | Model | Cost | Setup |
|---|---|---|---|
| Groq (default) | Llama 3.1 8B | Free tier | `GROQ_API_KEY` in `.env` |
| OpenAI | GPT-4o | Pay per use | Swap client in `llm_client.py` |
| Ollama (local) | Llama 3.2 3B | Free | No internet required |

---

## Requirements

```
Python >= 3.9
numpy, matplotlib, pandas, scipy
deap, ortools
groq, python-dotenv
jupyter (for notebook)
Node.js + docx (for Word report)
```

Install all Python dependencies:

```bash
pip install -r requirements.txt
```

---

## Test Coverage

```
Phase 1 — Problem definition    27 tests
Phase 2 — LLM + GA engine       34 tests
Phase 3 — Benchmarking          30 tests
Phase 4 — Visualization         21 tests
─────────────────────────────────────────
Total                           112 tests
```

Run specific phase:
```bash
pytest tests/test_phase1.py -v   # Phase 1 only
pytest tests/ -v                 # All phases
```

---

## Deliverables

| File | Description |
|---|---|
| `src/` | Full production codebase |
| `results/benchmark_results.csv` | All benchmark data |
| `results/plots/` | All charts (PNG) |
| `notebooks/ns_hybrid_demo.ipynb` | Live demo notebook |
| `client_report.docx` | Formal client report |

---

## References

- Taillard, E. (1993). Benchmarks for basic scheduling problems. *EJOR*, 64(2), 278–285.
- Fisher & Thompson (1963). Probabilistic learning combinations of local job-shop scheduling rules.
- Mao et al. (2024). Towards Cognitive AI Systems: A Survey on Neuro-Symbolic AI. arXiv:2401.01040.
- Giffler & Thompson (1960). Algorithms for solving production-scheduling problems. *Operations Research*.