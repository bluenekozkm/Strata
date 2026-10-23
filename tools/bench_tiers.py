#!/usr/bin/env python3
"""A/B the expert memory tiers on this machine: run the strata_bench suite per arm and read the engine's tier lines.

    python3 tools/bench_tiers.py --arm baseline --arm budget5

An arm is the installed config plus one mechanism under test (extra engine args, extra env).  Every arm gets its own
config copy under bench/arms/ (so its own engine log), runs tools/strata_bench.py against it on a bench port, and then
the full engine log is parsed for the tier lines (expert tiers, file tier I/O, decode hit rate) into the result
folder's tiers.json and a tiers.md summary.  `summary` prints every arm measured so far side by side.

    python3 tools/bench_tiers.py summary

The file-tier I/O switches are pinned to the same values in every arm (STRATA_IO_PREFETCH=1, STRATA_IO_PF_GATE=1,
STRATA_COLD_OVERLAP=4, STRATA_IO_STATS=1): the engine turns read-ahead on and off by page-cache room, and a budget
changes that room, so the arm would otherwise measure its own control state.

Storage note: the arms share the OS page cache and dropping it needs root, so run the weaker arm first and read each
arm's MemCached snapshot before comparing small differences.  Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS_DIR = ROOT / "bench" / "arms"
RESULTS_DIR = ROOT / "bench" / "results"
BENCH_PORT = 8077

# the same file-tier I/O behaviour in every arm, whatever the engine would pick per room
CONTROL_ENV = {
    "STRATA_IO_PREFETCH": "1",
    "STRATA_IO_PF_GATE": "1",
    "STRATA_COLD_OVERLAP": "4",
    "STRATA_IO_STATS": "1",
}

# the mechanisms under test; keep in step with docs/DETAILS.md (the resident budget section)
ARMS: dict[str, dict] = {
    "baseline": {"desc": "the installed config: --mmap-experts, no RAM copy"},
    "budget5": {"desc": "--resident-budget-gib 5 (the collapsed arm of the 2026-10-21 run)", "args": ["--resident-budget-gib", "5"]},
    "budget5-lendfirst": {"desc": "budget 5 with the prompt lend region admitted first (STRATA_BUDGET_LEND_FIRST=1)",
                          "args": ["--resident-budget-gib", "5"], "env": {"STRATA_BUDGET_LEND_FIRST": "1"}},
    "warmset6": {"desc": "no RAM copy, a 6 GiB managed page-cache warm set (STRATA_WARM_SET_GIB=6)",
                 "env": {"STRATA_WARM_SET_GIB": "6"}},
    "warmset8": {"desc": "no RAM copy, an 8 GiB managed page-cache warm set (STRATA_WARM_SET_GIB=8)",
                 "env": {"STRATA_WARM_SET_GIB": "8"}},
    "lendfirst-warm4": {"desc": "budget 5 with the lend region first plus a 4 GiB managed warm set",
                        "args": ["--resident-budget-gib", "5"],
                        "env": {"STRATA_BUDGET_LEND_FIRST": "1", "STRATA_WARM_SET_GIB": "4"}},
}


def meminfo() -> dict:
    out = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, _, rest = line.partition(":")
        out[key] = int(rest.strip().split()[0]) * 1024
    return out


def make_arm_config(base: dict, name: str) -> Path:
    arm = ARMS[name]
    cfg = json.loads(json.dumps(base))
    cfg["log"] = str(ARMS_DIR / f"{name}.log")
    env = dict(cfg.get("env") or {})
    env.update(CONTROL_ENV)
    env.update(arm.get("env") or {})
    cfg["env"] = env
    if arm.get("args"):
        cfg["args"] = list(cfg.get("args") or []) + list(arm["args"])
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    path = ARMS_DIR / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    return path


def run_arm(name: str, runs: int, port: int) -> Path:
    base = json.loads((ROOT / "strata-gyro-gyro-m.json").read_text(encoding="utf-8-sig"))
    cfg_path = make_arm_config(base, name)
    log_path = ARMS_DIR / f"{name}.log"
    if log_path.exists():
        log_path.unlink()
    mem0 = meminfo()
    t0 = time.time()
    cmd = [sys.executable, str(ROOT / "tools" / "strata_bench.py"), "--config", str(cfg_path),
           "--port", str(port), "--runs", str(runs), "--out", str(RESULTS_DIR)]
    print(f"[bench_tiers] arm {name}: {' '.join(cmd[2:])}", flush=True)
    done = subprocess.run(cmd, cwd=ROOT)
    if done.returncode != 0:
        sys.exit(f"[bench_tiers] arm {name}: strata_bench failed ({done.returncode})")
    folder = max((p for p in RESULTS_DIR.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime)
    metrics = parse_arm_log(log_path)
    metrics["arm"] = name
    metrics["desc"] = ARMS[name]["desc"]
    metrics["wall_s"] = round(time.time() - t0)
    metrics["mem_start"] = {k: mem0[k] for k in ("MemAvailable", "Cached", "MemFree") if k in mem0}
    mem1 = meminfo()
    metrics["mem_end"] = {k: mem1[k] for k in ("MemAvailable", "Cached", "MemFree") if k in mem1}
    (folder / "tiers.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    (folder / "tiers.md").write_text(render_tiers_md(metrics), encoding="utf-8")
    print(f"[bench_tiers] arm {name}: {folder} ({metrics['wall_s']} s)", flush=True)
    return folder


# ---- the engine log's tier lines -------------------------------------------------

RE = {
    "budget": re.compile(r"RAM budget ([\d.]+) GiB: (\d+) of the ([\d.]+) GiB"),
    "lend": re.compile(r"lend region: (\d+) of its (\d+) slots"),
    "file_reads": re.compile(r"the file tier reads (unbuffered|through the file cache) \(([^)]*)\)"),
    "resident_mode": re.compile(r"resident RAM mode: ([\d.]+) GiB of experts in RAM(?: \(page-locked\))?, (\d+) in the GPU cache"),
    "prefill_plan": re.compile(r"prefill plan inputs: ([^;]+); ([\d.]+) GiB available for the page cache"),
    "tiers": re.compile(r"expert tiers: GPU (\d+) hits this request; since the start RAM (\d+) blobs, files (\d+) blobs ([\d.]+) MB read"),
    "io": re.compile(r"file tier I/O this request: the OS read ([\d.]+) MB from storage \((\d+) major faults\) for ([\d.]+) MB of expert reads"),
    "cache_held": re.compile(r"page cache held ([\d.]+) MB at the read, not ([\d.]+) MB"),
    "crit": re.compile(r"decode waited ([\d.]+) ms in (\d+) fetches"),
    "hit": re.compile(r"decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)"),
    "moved": re.compile(r"(\d+) experts moved in from the file"),
}


def parse_arm_log(path: Path) -> dict:
    m: dict = {"startup": [], "requests": []}
    req: dict | None = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        for key in ("budget", "lend", "file_reads", "resident_mode", "prefill_plan"):
            hit = RE[key].search(line)
            if hit:
                m["startup"].append({"line": line.strip()[:400]})
        if (hit := RE["tiers"].search(line)) and "this request" in line:
            req = {"gpu_hits": int(hit.group(1)),
                   "file_blobs_total": int(hit.group(3)), "file_mb_total": float(hit.group(4))}
            m["requests"].append(req)
            continue
        if req is not None:
            if hit := RE["io"].search(line):
                req.update(storage_mb=float(hit.group(1)), faults=int(hit.group(2)), expert_mb=float(hit.group(3)))
                req["amplification"] = round(req["storage_mb"] / req["expert_mb"], 2) if req["expert_mb"] > 1 else None
                continue
            if hit := RE["cache_held"].search(line):
                req["cache_held_mb"] = float(hit.group(1))
                continue
            if hit := RE["crit"].search(line):
                req["crit_ms"] = float(hit.group(1))
                req["crit_fetches"] = int(hit.group(2))
                continue
            if hit := RE["hit"].search(line):
                req["hit_pct"] = float(hit.group(1))
                continue
            if "experts moved in from the file" in line and (hit := RE["moved"].search(line)):
                req["ram_adapt_moved"] = int(hit.group(1))
    return m


def med(vals: list) -> float | None:
    return round(statistics.median(vals), 2) if vals else None


def tier_summary(mt: dict) -> dict:
    reqs = mt.get("requests", [])
    tail = reqs[-6:] if len(reqs) > 6 else reqs
    storage = sum(r.get("storage_mb", 0.0) for r in reqs)
    expert = sum(r.get("expert_mb", 0.0) for r in reqs)
    return {
        "requests": len(reqs),
        "storage_gb_total": round(storage / 1e3, 2),
        "expert_read_gb_total": round(expert / 1e3, 2),
        "amplification": round(storage / expert, 2) if expert > 1 else None,
        "hit_pct_median_tail": med([r["hit_pct"] for r in tail if "hit_pct" in r]),
        "crit_ms_median_tail": med([r["crit_ms"] for r in tail if "crit_ms" in r]),
        "crit_ms_total": round(sum(r.get("crit_ms", 0.0) for r in reqs), 1),
        "ram_adapt_moved_total": sum(r.get("ram_adapt_moved", 0) for r in reqs),
        "lend_covered": next((int(RE["lend"].search(s["line"]).group(1)) for s in mt.get("startup", [])
                              if RE["lend"].search(s["line"])), None),
    }


def render_tiers_md(mt: dict) -> str:
    s = tier_summary(mt)
    L = [f"# Tier metrics: {mt['arm']} — {mt['desc']}", "",
         f"wall {mt['wall_s']} s; MemAvailable {mt['mem_start'].get('MemAvailable', 0) // (1 << 30)} GiB at the start, "
         f"{mt['mem_end'].get('MemAvailable', 0) // (1 << 30)} GiB at the end; page cache (Cached) "
         f"{mt['mem_start'].get('Cached', 0) // (1 << 30)} -> {mt['mem_end'].get('Cached', 0) // (1 << 30)} GiB.", "",
         "| metric | value |", "| --- | --- |"]
    for k, v in s.items():
        L.append(f"| {k} | {v} |")
    L += ["", "Startup lines:", ""] + [f"- `{s['line']}`" for s in mt.get("startup", [])[:12]]
    L += ["", "Per-request tail (the last 6):", "",
          "| gpu hits | hit % | storage MB | expert MB | amplification | decode waited ms |", "| --- | --- | --- | --- | --- | --- |"]
    for r in mt.get("requests", [])[-6:]:
        L.append(f"| {r.get('gpu_hits')} | {r.get('hit_pct')} | {r.get('storage_mb')} | {r.get('expert_mb')} | "
                 f"{r.get('amplification')} | {r.get('crit_ms')} |")
    return "\n".join(L) + "\n"


def bench_speeds(results: dict) -> dict:
    out = {}
    chat = results.get("chat", {}).get("requests", [])
    if chat:
        out["chat_decode"] = med([r["decode_tps"] for r in chat if r.get("decode_tps")])
    for lab in ("4K", "16K", "32K"):
        rs = results.get("prompts", {}).get(lab, {}).get("requests", [])
        if rs:
            out[f"{lab.lower()}_prompt"] = med([r["prompt_tps"] for r in rs if r.get("prompt_tps")])
    turns = results.get("agent", {}).get("turn_decode_tps") or {}
    vals = [v for vs in turns.values() for v in (vs if isinstance(vs, list) else [vs])]
    if vals:
        out["agent_decode"] = med(vals)
    return out


def summary() -> None:
    rows = []
    for tj in sorted(RESULTS_DIR.glob("*/tiers.json")):
        mt = json.loads(tj.read_text(encoding="utf-8"))
        res = tj.parent / "results.json"
        med_speed = bench_speeds(json.loads(res.read_text(encoding="utf-8")).get("results", {})) if res.exists() else {}
        rows.append((mt, tier_summary(mt), med_speed))
    if not rows:
        sys.exit("no tiers.json under bench/results yet")
    L = ["| arm | chat dec | 4K prompt | 16K prompt | agent dec | hit % | amplification | storage GB | crit ms (tail) | lend covered |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for mt, s, sp in rows:
        L.append(f"| {mt['arm']} | {sp.get('chat_decode')} | {sp.get('4k_prompt')} | {sp.get('16k_prompt')} | "
                 f"{sp.get('agent_decode')} | {s['hit_pct_median_tail']} | {s['amplification']} | "
                 f"{s['storage_gb_total']} | {s['crit_ms_median_tail']} | {s['lend_covered']} |")
    out = RESULTS_DIR / "tiers-summary.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", action="append", choices=sorted(ARMS), help="an arm to measure (repeatable, run in order)")
    ap.add_argument("--runs", type=int, default=3, help="strata_bench runs per test (default 3)")
    ap.add_argument("--port", type=int, default=BENCH_PORT, help="the port the arm's server starts on")
    ap.add_argument("cmd", nargs="?", choices=["summary"], help="print the arms measured so far side by side")
    a = ap.parse_args()
    if a.cmd == "summary":
        summary()
    elif a.arm:
        for name in a.arm:
            run_arm(name, a.runs, a.port)
        summary()
    else:
        ap.error("give --arm (repeatable) or summary")


if __name__ == "__main__":
    main()
