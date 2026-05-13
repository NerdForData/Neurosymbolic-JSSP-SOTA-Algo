"""
Benchmark Instance Loader
=========================
Loads JSSP benchmark instances from:
  1. Built-in data (FT06, FT10)     -- always available, no download
  2. Local cache  (instances/)       -- after first download
  3. Auto-download from JSPLIB       -- on first request (requires internet)

Supported instance families:
  - Fisher & Thompson : ft06, ft10, ft20
  - Taillard          : ta01 .. ta80
  - DMU               : dmu01 .. dmu80

Usage:
    from src.problem.loader import load_instance, download_all_tier

    inst = load_instance("ft06")          # built-in, instant
    inst = load_instance("ta01")          # downloads on first call, cached after
    download_all_tier(tier=2)             # batch-download Tier 2 (ta01-ta30)
"""

import os
import urllib.request
import urllib.error
import time
from pathlib import Path
from typing import Optional
from src.problem.jssp import JSSPInstance, Job, Machine, Operation


# ---------------------------------------------------------------------------
# Best-known solutions from published literature
# ---------------------------------------------------------------------------

BEST_KNOWN = {
    # Fisher & Thompson
    "ft06": 55,
    "ft10": 930,
    "ft20": 1165,
    # Taillard Ta01-Ta10  (15 jobs x 15 machines)
    "ta01": 1231, "ta02": 1244, "ta03": 1218, "ta04": 1175, "ta05": 1224,
    "ta06": 1238, "ta07": 1227, "ta08": 1217, "ta09": 1274, "ta10": 1241,
    # Taillard Ta11-Ta20  (20 jobs x 15 machines)
    "ta11": 1357, "ta12": 1367, "ta13": 1342, "ta14": 1345, "ta15": 1339,
    "ta16": 1360, "ta17": 1462, "ta18": 1369, "ta19": 1375, "ta20": 1345,
    # Taillard Ta21-Ta30  (20 jobs x 20 machines)
    "ta21": 1644, "ta22": 1600, "ta23": 1557, "ta24": 1644, "ta25": 1595,
    "ta26": 1643, "ta27": 1680, "ta28": 1603, "ta29": 1625, "ta30": 1584,
    # Taillard Ta31-Ta40  (30 jobs x 15 machines)
    "ta31": 1764, "ta32": 1796, "ta33": 1788, "ta34": 1853, "ta35": 2001,
    "ta36": 1819, "ta37": 1829, "ta38": 1673, "ta39": 1655, "ta40": 1764,
    # Taillard Ta41-Ta50  (30 jobs x 20 machines)
    "ta41": 2018, "ta42": 1884, "ta43": 1809, "ta44": 1948, "ta45": 1997,
    "ta46": 2011, "ta47": 1807, "ta48": 1953, "ta49": 1960, "ta50": 1923,
    # Taillard Ta51-Ta60  (50 jobs x 15 machines)
    "ta51": 2760, "ta52": 2756, "ta53": 2717, "ta54": 2839, "ta55": 2679,
    "ta56": 2781, "ta57": 2943, "ta58": 2885, "ta59": 2765, "ta60": 2655,
    # Taillard Ta61-Ta70  (50 jobs x 20 machines)
    "ta61": 3045, "ta62": 2867, "ta63": 2825, "ta64": 2982, "ta65": 2997,
    "ta66": 3055, "ta67": 3010, "ta68": 3046, "ta69": 3038, "ta70": 2926,
    # Taillard Ta71-Ta80  (100 jobs x 20 machines)
    "ta71": 5464, "ta72": 5181, "ta73": 5568, "ta74": 5339, "ta75": 5392,
    "ta76": 5342, "ta77": 5436, "ta78": 5394, "ta79": 5358, "ta80": 5183,
}

# Download URLs -- JSPLIB is the canonical public GitHub mirror of OR-Library instances
_BASE_URL = "https://raw.githubusercontent.com/tamy0612/JSPLIB/master/instances"

INSTANCE_URLS = {
    "ft06": f"{_BASE_URL}/ft06.txt",
    "ft10": f"{_BASE_URL}/ft10.txt",
    "ft20": f"{_BASE_URL}/ft20.txt",
    **{f"ta{i:02d}": f"{_BASE_URL}/ta{i:02d}.txt" for i in range(1, 81)},
    **{f"dmu{i:02d}": f"{_BASE_URL}/DMU/dmu{i:02d}.txt" for i in range(1, 81)},
}

# Tier definitions for batch downloads
TIERS = {
    1: ["ft06", "ft10", "ft20"],
    2: [f"ta{i:02d}" for i in range(1, 31)],
    3: [f"ta{i:02d}" for i in range(31, 81)],
}


# ---------------------------------------------------------------------------
# Built-in instances (no download needed)
# ---------------------------------------------------------------------------

_FT06 = """\
6 6
2 1  0 3  1 6  3 7  5 3  4 6
1 8  2 5  4 10 5 10 0 10 3 4
2 5  3 4  5 8  0 9  1 1  4 7
1 5  0 5  2 5  3 3  4 8  5 9
2 9  1 3  4 5  5 4  0 3  3 1
1 3  3 3  5 9  0 10 4 4  2 1"""

_FT10 = """\
10 10
0 29 1 78 2 9  3 36 4 49 5 11 6 62 7 56 8 44 9 21
0 43 2 90 4 75 9 11 3 69 1 28 6 46 5 46 7 72 8 30
1 91 0 85 3 39 2 74 8 90 5 10 7 12 6 89 9 45 4 33
1 81 2 95 0 71 4 99 6 9  8 52 7 85 3 98 9 22 5 43
2 14 0 6  1 22 5 61 3 26 4 69 8 21 6 49 9 72 7 53
2 84 1 2  5 52 3 95 8 48 9 72 0 47 6 65 4 6  7 25
1 46 0 37 3 61 2 13 6 32 5 21 9 32 8 89 7 30 4 55
2 31 0 86 1 46 5 74 4 32 6 88 8 19 9 48 7 36 3 79
0 76 1 69 3 76 5 51 2 85 9 11 6 40 7 89 4 26 8 74
1 85 0 13 2 61 6 7  8 64 9 76 5 47 3 52 4 90 7 45"""

_BUILTIN = {
    "ft06": _FT06,
    "ft10": _FT10,
}


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _get_weights(n_jobs: int, n_machines: int) -> dict:
    """Scale objective weights based on instance size."""
    if n_jobs <= 10:
        return dict(w_makespan=0.40, w_tardiness=0.25, w_utilization=0.15,
                    w_flowtime=0.12, w_energy=0.08)
    elif n_jobs <= 30:
        return dict(w_makespan=0.35, w_tardiness=0.25, w_utilization=0.15,
                    w_flowtime=0.15, w_energy=0.10)
    else:
        return dict(w_makespan=0.30, w_tardiness=0.20, w_utilization=0.20,
                    w_flowtime=0.15, w_energy=0.15)


def _parse(text: str, name: str) -> JSSPInstance:
    """
    Parse standard OR-Library / Taillard / JSPLIB format.

    Format:
        Line 1 : n_jobs  n_machines
        Lines 2+: space-separated pairs (machine_id  processing_time) per job
    Comment lines starting with # are ignored.
    """
    lines = [
        l.strip() for l in text.strip().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]

    # First line: dimensions
    dims = lines[0].split()
    n_jobs, n_machines = int(dims[0]), int(dims[1])

    machines = [Machine(machine_id=m) for m in range(n_machines)]
    jobs = []
    total_proc = 0

    for j_idx in range(n_jobs):
        tokens = lines[1 + j_idx].split()
        ops = []
        for step in range(n_machines):
            m_id   = int(tokens[2 * step])
            p_time = int(tokens[2 * step + 1])
            ops.append(Operation(
                job_id=j_idx,
                step=step,
                machine_id=m_id,
                processing_time=p_time,
            ))
            total_proc += p_time
        jobs.append(Job(job_id=j_idx, operations=ops))

    # Synthetic due dates: 1.3x lower-bound makespan (standard practice)
    lb = total_proc // n_machines
    for job in jobs:
        job.due_date = int(lb * 1.3)

    return JSSPInstance(
        name=name,
        jobs=jobs,
        machines=machines,
        best_known_solution=BEST_KNOWN.get(name.lower()),
        **_get_weights(n_jobs, n_machines),
    )


# ---------------------------------------------------------------------------
# Downloader
# ---------------------------------------------------------------------------

def _download_raw(name: str, data_dir: str = "instances") -> str:
    """
    Download instance from JSPLIB GitHub mirror and cache locally.
    Returns the raw text content.
    """
    name_lower = name.lower()
    url = INSTANCE_URLS.get(name_lower)
    if not url:
        raise ValueError(
            f"Unknown instance '{name}'. "
            f"Supported: ft06/ft10/ft20, ta01-ta80, dmu01-dmu80."
        )

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    cache_path = Path(data_dir) / f"{name_lower}.txt"

    print(f"  Downloading {name_lower} from JSPLIB...")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nsai-jss/1.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            content = response.read().decode("utf-8")
        cache_path.write_text(content)
        print(f"  Saved to {cache_path}")
        return content
    except urllib.error.URLError as e:
        raise ConnectionError(
            f"Could not download '{name}' from {url}.\n"
            f"Error: {e}\n"
            f"Manually download and place at: {cache_path}"
        ) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_instance(name: str, data_dir: str = "instances") -> JSSPInstance:
    """
    Load a JSSP benchmark instance by name.

    Resolution order:
      1. Built-in (ft06, ft10)    -- instant, no I/O
      2. Local cache (data_dir/)  -- fast, from previous download
      3. Auto-download from JSPLIB GitHub mirror

    Args:
        name:     Instance name e.g. 'ft06', 'ta01', 'ta30', 'dmu05'
        data_dir: Directory for caching downloaded instances

    Returns:
        JSSPInstance ready for use
    """
    key = name.lower()

    # 1. Built-in
    if key in _BUILTIN:
        return _parse(_BUILTIN[key], key)

    # 2. Local cache
    cache_path = Path(data_dir) / f"{key}.txt"
    if cache_path.exists():
        return _parse(cache_path.read_text(), key)

    # 3. Download
    content = _download_raw(key, data_dir)
    return _parse(content, key)


def download_instance(name: str, data_dir: str = "instances") -> JSSPInstance:
    """Force download (or re-download) an instance, ignoring local cache."""
    key = name.lower()
    content = _download_raw(key, data_dir)
    return _parse(content, key)


def download_all_tier(tier: int, data_dir: str = "instances", delay: float = 0.3):
    """
    Download all instances in a benchmark tier.

    Args:
        tier:     1 = small (ft06/ft10/ft20)
                  2 = medium (ta01-ta30)
                  3 = large  (ta31-ta80)
        data_dir: Local cache directory
        delay:    Seconds between requests (be polite to the server)
    """
    instances = TIERS.get(tier)
    if not instances:
        raise ValueError(f"Unknown tier {tier}. Choose 1, 2, or 3.")

    print(f"Downloading Tier {tier} ({len(instances)} instances) into '{data_dir}/'")
    ok, failed = [], []

    for name in instances:
        key = name.lower()
        cache_path = Path(data_dir) / f"{key}.txt"

        # Skip if already cached
        if key in _BUILTIN:
            print(f"  {key}: built-in, skipping download")
            ok.append(key)
            continue
        if cache_path.exists():
            print(f"  {key}: already cached, skipping")
            ok.append(key)
            continue

        try:
            _download_raw(key, data_dir)
            ok.append(key)
            time.sleep(delay)
        except Exception as e:
            print(f"  {key}: FAILED -- {e}")
            failed.append(key)

    print(f"\nDone. {len(ok)} downloaded/cached, {len(failed)} failed.")
    if failed:
        print(f"Failed: {failed}")
        print("You can manually download from: https://github.com/tamy0612/JSPLIB/tree/master/instances")


def list_instances():
    """Print a summary of all supported benchmark instances."""
    print("=" * 60)
    print("  JSSP Benchmark Instances")
    print("=" * 60)
    rows = [
        ("Tier 1 — Small",  "ft06, ft10, ft20",         "6-20 jobs,  known optimum"),
        ("Tier 2 — Medium", "ta01 .. ta30",              "15-20 jobs, BKS published"),
        ("Tier 3 — Large",  "ta31 .. ta80",              "30-100 jobs, exact solvers fail"),
        ("Extra",           "dmu01 .. dmu80",            "mixed-size DMU instances"),
    ]
    for label, names, note in rows:
        print(f"\n{label}")
        print(f"  Instances : {names}")
        print(f"  Note      : {note}")
    print()