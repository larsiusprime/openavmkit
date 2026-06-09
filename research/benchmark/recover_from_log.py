"""Recover per-model benchmark tables from a run log.

The benchmark comparison tables are printed during a run (``MAIN Benchmark (<group>)`` sections,
each with a ``Holdout set:`` and ``Study set:`` transposed-markdown table). When a run's compute
succeeded but the in-process persistence missed them, this recovers the real numbers from the log
without re-fitting -- it parses those printed tables into tidy per-group CSVs (models as rows, stats
as columns) plus a combined ``benchmark.md``.

Usage:
    python research/benchmark/recover_from_log.py us-va-petersburgcity research/benchmark/petersburg_run.log
"""
import os
import re
import sys

import pandas as pd


def _parse_md_table(lines, start):
    """Parse a markdown table starting at ``lines[start]`` (the header row). Returns (df, next_idx).

    The printed tables are *transposed*: header = model names, each subsequent row = one statistic.
    Returns a DataFrame indexed by model with one column per statistic.
    """
    def cells(row):
        return [c.strip() for c in row.strip().strip("|").split("|")]

    header = cells(lines[start])          # ['', 'assessor', 'mra', ...]
    models = header[1:]
    i = start + 2                          # skip the |:---| separator row
    stats = {}
    while i < len(lines) and lines[i].lstrip().startswith("|"):
        row = cells(lines[i])
        stat = row[0]
        if stat:
            stats[stat] = row[1:]
        i += 1
    # build model-indexed frame
    data = {m: {} for m in models}
    for stat, vals in stats.items():
        for m, v in zip(models, vals):
            data[m][stat] = v
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "model"
    return df, i


def recover(slug, log_path, keep_groups=None, out_root="research/benchmark"):
    lines = open(log_path, encoding="utf-8", errors="replace").read().splitlines()
    out_dir = os.path.join(out_root, slug)
    keep = set(keep_groups) if keep_groups else None
    groups = []
    cur_group = None
    i = 0
    pending = {}  # group -> {subset: df}
    while i < len(lines):
        m = re.match(r"MAIN Benchmark \(([^)]+)\) -- Assessor Metrics", lines[i])
        if m:
            cur_group = m.group(1)
            pending.setdefault(cur_group, {})
            i += 1
            continue
        # only capture the Assessor-Metrics block (stop at Academic Metrics)
        if cur_group and re.match(r"MAIN Benchmark \(.*\) -- Academic", lines[i]):
            cur_group = None
            i += 1
            continue
        if cur_group:
            label = lines[i].strip().rstrip(":")
            if label in ("Holdout set", "Study set") and i + 1 < len(lines) and lines[i + 1].lstrip().startswith("|"):
                df, nxt = _parse_md_table(lines, i + 1)
                pending[cur_group][label] = df
                i = nxt
                continue
        i += 1

    for group, subsets in pending.items():
        if not subsets:
            continue
        if keep is not None and group not in keep:
            continue
        g_dir = os.path.join(out_dir, group)
        os.makedirs(g_dir, exist_ok=True)
        parts = []
        for label, fname in [("Holdout set", "benchmark_holdout.csv"),
                             ("Study set", "benchmark_study.csv")]:
            if label in subsets:
                subsets[label].to_csv(os.path.join(g_dir, fname))
                parts.append(f"### {label}\n\n{subsets[label].to_markdown()}\n")
        with open(os.path.join(g_dir, "benchmark.md"), "w", encoding="utf-8") as f:
            f.write(f"# Benchmark — {slug} / {group}\n\n")
            f.write("Recovered from run log (`recover_from_log.py`). Per-model comparison: "
                    "utility_score, counts, median_ratio, COD/PRD/PRB/VEI (+trimmed), CHD. "
                    "Holdout = test set; Study = universe/full set.\n\n")
            f.write("\n".join(parts))
        groups.append(group)
        print(f"[{slug}] recovered {group}: {list(subsets.keys())}")
    return groups


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: recover_from_log.py <slug> <log_path> [keep_group ...]")
    keep = sys.argv[3:] or None
    recover(sys.argv[1], sys.argv[2], keep_groups=keep)
