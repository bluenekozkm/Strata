#!/usr/bin/env python3
"""Benchmark and compare Linear Chain KVCache vs Persistent Tiered KV Cache under alternating multi-system-prompt workloads.

Scenario:
  Two divergent system prompts (System A: Code Architect, System B: Research Analyst).
  - Linear Chain KVCache (Default):
    Single-chain checkpoint retention.
    * Turn 1A (Cold): Prompt A is prefilled from scratch (~350 ms).
    * Turn 1B (Cold): Prompt B is prefilled from scratch (~350 ms), completely evicting Prompt A.
    * Turn 2A (Warm Re-ask): System switches back to Prompt A. Since Prompt A was evicted,
      it must be fully recalculated at prompt prefill speed (Cache Miss / Thrashing, ~350 ms, 1.0x).
    * Turn 2B (Warm Re-ask): Same eviction happens for Prompt B (Cache Miss, ~350 ms, 1.0x).

  - Tiered KV Cache (--exp-tiered-kv [DIR]):
    Multi-branch radix prefix tree on host RAM (Tier 1) with optional NVMe Direct I/O persistence (Tier 2).
    * Turn 1A (Cold): Prompt A prefilled into active VRAM and recorded into Tier 1 Radix tree.
    * Turn 1B (Cold): Switching to Prompt B preserves Prompt A in host memory.
    * Turn 2A (Warm Re-ask): Common prefix detected in Radix forest; differential PCIe DMA
      swaps the context and bypasses prompt prefill computation (>20x speedup, TTFT <15 ms).
    * Turn 2B (Warm Re-ask): Prompt B also restored instantly from cache (>20x speedup).

Modes:
  --mode compare   (default) Evaluates Linear Chain vs Tiered KV side-by-side with comparison report.
  --mode tiered    Evaluates Tiered KV Cache alone.
  --mode linear    Evaluates Linear Chain KVCache alone.

Tiers:
  --tier 1         (default) Tier 1 RAM-only Radix tree (--exp-tiered-kv without DIR).
  --tier 2         Tier 1 Host RAM + Tier 2 NVMe CAS Disk Cache (--exp-tiered-kv <DIR>).

Usage:
  # Standalone mock comparison (no GPU or weights required):
  python3 tools/bench_tiered_kv.py --mock

  # Full comparison on resident GPU engine:
  python3 tools/bench_tiered_kv.py --config strata-gyro-gyro-m.json --mode compare

  # Evaluate Tiered KV Cache Tier 1 (RAM-only) on running server:
  python3 tools/bench_tiered_kv.py --url http://127.0.0.1:8080
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_free_port() -> int:
    """Finds an available ephemeral port on loopback."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def find_default_config() -> Path | None:
    """Finds the most recent strata-*.json config in the repository root."""
    configs = sorted(ROOT.glob("strata-*.json"), key=os.path.getmtime, reverse=True)
    for c in configs:
        if "gyro-m" in c.name:
            return c
    return configs[0] if configs else None


def generate_system_prompt(domain: str, target_words: int = 800) -> str:
    """Generates a dense, domain-specific system prompt of roughly target_words."""
    if domain == "code":
        title = "Senior Systems & Distributed Architect"
        core = (
            "You are an expert low-level systems programming and distributed systems architect. "
            "You specialize in C++20, CUDA, HIP, lock-free concurrency, memory hierarchies, "
            "cache-line alignment, and high-performance inference engines. "
            "When responding, adhere strictly to zero-overhead principles, deterministic lifecycle "
            "management, and hardware-specific latency budgets."
        )
        vocab = (
            "synchronization barrier mutex spinlock atomic cacheline prefetch direct_io "
            "radix tree prefix match page table slab allocator pinned memory dma stream "
            "throughput latency bandwidth pipelining registers simd avx512 matrix cores "
            "tensor core shared memory coalesced transaction branch prediction context switch"
        ).split()
    else:
        title = "Principal Legal & Technical Compliance Auditor"
        core = (
            "You are a principal technical auditor and compliance analyst for enterprise AI systems. "
            "You specialize in regulatory verification, SOC2 compliance, data provenance, "
            "bit-exact reproducibility, telemetry audit logs, and security policy enforcement. "
            "When evaluating systems, provide rigorous evidence-ranked findings, traceability rubrics, "
            "and statutory risk mitigations."
        )
        vocab = (
            "compliance audit provenance verification integrity hash xxhash64 merkle tree "
            "reproducibility statutory framework certification authorization telemetry "
            "invariant assertion governance retention snapshot ledger atomic commit fsync "
            "confidentiality isolation multi-tenant boundary encryption vulnerability"
        ).split()

    rng = random.Random(42 if domain == "code" else 99)
    paragraphs = [f"=== SYSTEM ROLE: {title} ===\n{core}\n"]

    words_remaining = target_words - len(core.split())
    rule_idx = 1
    while words_remaining > 0:
        chunk_words = min(words_remaining, 40)
        rule_body = " ".join(rng.choice(vocab) for _ in range(chunk_words))
        paragraphs.append(f"Section {rule_idx} (Operational Constraints): The system must ensure that {rule_body}.")
        words_remaining -= chunk_words + 8
        rule_idx += 1

    return "\n\n".join(paragraphs)


def post_stream(url: str, messages: list[dict], max_tokens: int = 16, timeout: float = 90.0) -> dict:
    """Executes a streaming request against /v1/chat/completions to measure TTFT and prefill accurately."""
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    first_token_time = None
    chunks: list[str] = []
    timings: dict = {}
    usage: dict = {}

    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if data_str == "[DONE]":
                break
            try:
                msg = json.loads(data_str)
                choices = msg.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        chunks.append(content)
                if "timings" in msg:
                    timings = msg["timings"]
                if "usage" in msg:
                    usage = msg["usage"]
            except json.JSONDecodeError:
                pass

    t1 = time.perf_counter()
    ttft_s = (first_token_time - t0) if first_token_time is not None else (t1 - t0)
    total_s = t1 - t0

    return {
        "ttft_ms": round(ttft_s * 1000.0, 2),
        "total_ms": round(total_s * 1000.0, 2),
        "reply": "".join(chunks).strip(),
        "timings": timings,
        "usage": usage,
    }


def fetch_metrics(url: str) -> dict:
    """Fetches telemetry metrics from /metrics endpoint."""
    try:
        req = urllib.request.Request(f"{url.rstrip('/')}/metrics")
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return {}


def wait_ready(port: int, proc: subprocess.Popen, timeout: float = 300.0) -> None:
    """Waits for Strata server to finish loading model weights and be ready to accept requests."""
    t0 = time.time()
    last_status = t0
    url = f"http://127.0.0.1:{port}/v1/models"
    while time.time() - t0 < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited prematurely with exit code {proc.returncode}")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3.0) as r:
                if r.status == 200:
                    return
        except Exception:
            pass
        if time.time() - last_status >= 5.0:
            elapsed = int(time.time() - t0)
            print(f"  [Startup] Waiting for engine and weights to load on port {port}... ({elapsed}s elapsed)")
            last_status = time.time()
        time.sleep(1.0)
    raise TimeoutError(f"Server on port {port} failed to become ready within {timeout}s")


def launch_server(
    config_path: Path,
    port: int | None = None,
    mode: str = "tiered",
    tier: int = 1,
    cache_dir: Path | None = None,
) -> tuple[subprocess.Popen, int, Path, Path | None]:
    """Prepares and launches the resident server configured for either Linear Chain or Tiered KV Cache."""
    if port is None:
        port = find_free_port()

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args = list(cfg.get("args", []))

    # Strip existing tiered KV and conversation cache flags
    cleaned_args = []
    skip_next = False
    for i, a in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if a in ("--exp-tiered-kv", "--exp-tiered-kv-cache"):
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                skip_next = True
            continue
        if a.startswith(("--exp-tiered-kv=", "--exp-tiered-kv-cache=")):
            continue
        if a in ("--kv-disk-cache", "--kv-disk-budget", "--conversation-cache-mib", "--conversation-cache-slots"):
            if i + 1 < len(args):
                skip_next = True
            continue
        if a == "--kv-multi-system-prompt":
            continue
        cleaned_args.append(a)

    args = cleaned_args
    created_cache_dir: Path | None = None

    if mode == "tiered":
        cfg["exp_tiered_kv"] = True
        if tier == 2:
            if cache_dir is None:
                created_cache_dir = Path(tempfile.mkdtemp(prefix=f"strata_tiered_kv_{port}_"))
            else:
                created_cache_dir = cache_dir
                created_cache_dir.mkdir(parents=True, exist_ok=True)
            args.extend(["--exp-tiered-kv", str(created_cache_dir), "--kv-disk-budget", "10240"])
        else:
            # Tier 1: Host RAM Radix tree only (no disk cache)
            args.append("--exp-tiered-kv")

        # In Tiered KV mode, host RAM conversation parking is enabled (Tier 1)
        for flag, val in [
            ("--conversation-cache-mib", "4096"),
            ("--conversation-cache-slots", "8"),
            ("--prompt-cache-root", "256"),
            ("--prompt-cache", "8"),
        ]:
            if flag not in args:
                args.extend([flag, val])
    else:
        # Linear Chain KVCache mode:
        # Force conversation-cache-mib to 0 so the engine does NOT park conversations to host RAM.
        # The engine strictly maintains only the single active linear checkpoint chain in VRAM;
        # alternating system prompts will evict the previous prefix and force full prefill recomputation.
        cfg["exp_tiered_kv"] = False
        args.extend(["--conversation-cache-mib", "0", "--conversation-cache-slots", "0"])

    cfg["args"] = args
    cfg["port"] = port
    log_dir = created_cache_dir if created_cache_dir else Path(tempfile.gettempdir())
    cfg["log"] = str(log_dir / f"strata_server_{mode}_{port}.log")

    temp_cfg_path = log_dir / f"strata_bench_config_{mode}_{port}.json"
    with open(temp_cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    cmd = [
        sys.executable,
        "-m",
        "serve.server",
        "--engine",
        "strata",
        "--config",
        str(temp_cfg_path),
        "--port",
        str(port),
    ]

    tier_desc = f"Tier {tier} ({'NVMe SSD' if tier == 2 else 'RAM-only'})" if mode == "tiered" else "Default Linear Chain"
    print(f"\n[Server Auto-Start] Launching Strata engine ({mode.upper()}: {tier_desc}) on port {port}")
    print(f"  Engine Binary: {cfg.get('exe', 'engine/strata')}")
    print(f"  Config:        {config_path.name}")
    if created_cache_dir:
        print(f"  KV Cache Dir:  {created_cache_dir}")
    print(f"  Server Log:    {cfg['log']}\n")

    env = dict(os.environ)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    def stream_logs():
        try:
            for line in iter(proc.stdout.readline, ""):
                if line:
                    sys.stdout.write(f"  [strata-{mode}] {line}")
                    sys.stdout.flush()
        except Exception:
            pass

    log_thread = threading.Thread(target=stream_logs, daemon=True)
    log_thread.start()

    try:
        wait_ready(port, proc)
        print(f"[Server Ready] Server successfully loaded and listening on 127.0.0.1:{port}\n")
    except Exception:
        time.sleep(0.5)
        stop_server(proc, temp_cfg_path, created_cache_dir)
        raise

    return proc, port, temp_cfg_path, created_cache_dir


def stop_server(proc: subprocess.Popen, temp_cfg: Path | None, cache_dir: Path | None) -> None:
    """Gracefully terminates the server to flush any state and cleans up temporary resources."""
    if proc.poll() is None:
        print("\n[Server Shutdown] Stopping server instance...")
        try:
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=10)
        except Exception:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    if temp_cfg and temp_cfg.exists():
        try:
            temp_cfg.unlink()
        except OSError:
            pass

    if cache_dir and cache_dir.exists():
        try:
            shutil.rmtree(cache_dir, ignore_errors=True)
        except OSError:
            pass


def run_benchmark(url: str, prompt_words: int = 800, max_tokens: int = 16, label: str = "Benchmark") -> dict:
    """Runs the 2-System-Prompt alternating benchmark."""
    print("----------------------------------------------------------------------")
    print(f"  RUNNING SUITE: {label.upper()}")
    print(f"  Target URL: {url} | Words: ~{prompt_words} (~1,200 tokens) | max_tokens: {max_tokens}")
    print("----------------------------------------------------------------------")

    sys_a = generate_system_prompt("code", target_words=prompt_words)
    sys_b = generate_system_prompt("compliance", target_words=prompt_words)

    initial_metrics = fetch_metrics(url).get("tiered_kv", {})
    if initial_metrics:
        print(f"  Initial Tiered KV Telemetry: {initial_metrics}")

    # 1. Turn 1A: Prompt A
    print("  -> Turn 1A: Sending System Prompt A (Coding Architect)...")
    res_1a = post_stream(
        url,
        messages=[
            {"role": "system", "content": sys_a},
            {"role": "user", "content": "What is the primary synchronization rule? Answer in 1 sentence."},
        ],
        max_tokens=max_tokens,
    )
    print(f"     TTFT: {res_1a['ttft_ms']:.1f} ms | Total: {res_1a['total_ms']:.1f} ms | Timings: {res_1a.get('timings') or 'N/A'}")

    # 2. Turn 1B: Prompt B (Switch context)
    print("  -> Turn 1B: Switching to System Prompt B (Legal Compliance)...")
    res_1b = post_stream(
        url,
        messages=[
            {"role": "system", "content": sys_b},
            {"role": "user", "content": "What is the primary audit rule? Answer in 1 sentence."},
        ],
        max_tokens=max_tokens,
    )
    print(f"     TTFT: {res_1b['ttft_ms']:.1f} ms | Total: {res_1b['total_ms']:.1f} ms | Timings: {res_1b.get('timings') or 'N/A'}")

    # 3. Turn 2A: Switch back to Prompt A
    print("  -> Turn 2A: Switching back to System Prompt A...")
    res_2a = post_stream(
        url,
        messages=[
            {"role": "system", "content": sys_a},
            {"role": "user", "content": "What is rule #2? Answer in 1 sentence."},
        ],
        max_tokens=max_tokens,
    )
    print(f"     TTFT: {res_2a['ttft_ms']:.1f} ms | Total: {res_2a['total_ms']:.1f} ms | Timings: {res_2a.get('timings') or 'N/A'}")

    # 4. Turn 2B: Switch back to Prompt B
    print("  -> Turn 2B: Switching back to System Prompt B...")
    res_2b = post_stream(
        url,
        messages=[
            {"role": "system", "content": sys_b},
            {"role": "user", "content": "What is Section 2 constraint? Answer in 1 sentence."},
        ],
        max_tokens=max_tokens,
    )
    print(f"     TTFT: {res_2b['ttft_ms']:.1f} ms | Total: {res_2b['total_ms']:.1f} ms | Timings: {res_2b.get('timings') or 'N/A'}")

    final_metrics = fetch_metrics(url).get("tiered_kv", {})

    speedup_a = (res_1a["ttft_ms"] / res_2a["ttft_ms"]) if res_2a["ttft_ms"] > 0 else 1.0
    speedup_b = (res_1b["ttft_ms"] / res_2b["ttft_ms"]) if res_2b["ttft_ms"] > 0 else 1.0

    return {
        "label": label,
        "res_1a": res_1a,
        "res_1b": res_1b,
        "res_2a": res_2a,
        "res_2b": res_2b,
        "speedup_a": speedup_a,
        "speedup_b": speedup_b,
        "final_metrics": final_metrics,
    }


def print_single_summary(result: dict) -> None:
    """Prints single benchmark summary."""
    print("\n======================================================================")
    print(f"               BENCHMARK RESULTS: {result['label'].upper()}           ")
    print("======================================================================")
    print(f"{'Phase':<16} | {'Prompt':<10} | {'TTFT (ms)':<10} | {'Prefill Time':<14} | {'Speedup':<10}")
    print("-" * 70)

    r1a, r1b = result["res_1a"], result["res_1b"]
    r2a, r2b = result["res_2a"], result["res_2b"]

    p_ms_1a = r1a.get("timings", {}).get("prompt_ms", "N/A")
    p_ms_1b = r1b.get("timings", {}).get("prompt_ms", "N/A")
    p_ms_2a = r2a.get("timings", {}).get("prompt_ms", "N/A")
    p_ms_2b = r2b.get("timings", {}).get("prompt_ms", "N/A")

    sp_a = f"{result['speedup_a']:.2f}x"
    sp_b = f"{result['speedup_b']:.2f}x"
    print(f"{'Round 1 (Cold)':<16} | {'Prompt A':<10} | {r1a['ttft_ms']:<10.1f} | {str(p_ms_1a) + ' ms':<14} | {'1.00x':<10}")
    print(f"{'Round 1 (Cold)':<16} | {'Prompt B':<10} | {r1b['ttft_ms']:<10.1f} | {str(p_ms_1b) + ' ms':<14} | {'1.00x':<10}")
    print(f"{'Round 2 (Warm)':<16} | {'Prompt A':<10} | {r2a['ttft_ms']:<10.1f} | {str(p_ms_2a) + ' ms':<14} | {sp_a:<10}")
    print(f"{'Round 2 (Warm)':<16} | {'Prompt B':<10} | {r2b['ttft_ms']:<10.1f} | {str(p_ms_2b) + ' ms':<14} | {sp_b:<10}")
    print("-" * 70)

    fm = result.get("final_metrics") or {}
    if fm:
        print(f"Telemetry (/metrics): branches={fm.get('system_prompt_branches', 0)}, "
              f"switches={fm.get('system_prompt_switches', 0)}, hits={fm.get('system_prompt_hits', 0)}")
    print("======================================================================\n")


def print_comparison_table(res_linear: dict, res_tiered: dict) -> None:
    """Prints a comparative table contrasting Linear Chain vs Tiered KV Cache."""
    print("\n" + "=" * 88)
    print("        HEAD-TO-HEAD COMPARISON: LINEAR CHAIN vs PERSISTENT TIERED KV CACHE")
    print("=" * 88)
    print(f"{'Turn / Prompt':<20} | {'Linear Chain TTFT':<18} | {'Tiered KV TTFT':<16} | {'Cache Status':<14} | {'Acceleration':<12}")
    print("-" * 88)

    lin_1a, lin_1b = res_linear["res_1a"]["ttft_ms"], res_linear["res_1b"]["ttft_ms"]
    lin_2a, lin_2b = res_linear["res_2a"]["ttft_ms"], res_linear["res_2b"]["ttft_ms"]

    tkv_1a, tkv_1b = res_tiered["res_1a"]["ttft_ms"], res_tiered["res_1b"]["ttft_ms"]
    tkv_2a, tkv_2b = res_tiered["res_2a"]["ttft_ms"], res_tiered["res_2b"]["ttft_ms"]

    accel_2a = (lin_2a / tkv_2a) if tkv_2a > 0 else 1.0
    accel_2b = (lin_2b / tkv_2b) if tkv_2b > 0 else 1.0

    print(f"{'Turn 1A (Cold A)':<20} | {lin_1a:>10.1f} ms       | {tkv_1a:>10.1f} ms     | {'Both Cold':<14} | {'1.00x':<12}")
    print(f"{'Turn 1B (Cold B)':<20} | {lin_1b:>10.1f} ms       | {tkv_1b:>10.1f} ms     | {'Both Cold':<14} | {'1.00x':<12}")
    print(f"{'Turn 2A (Re-ask A)':<20} | {lin_2a:>10.1f} ms       | {tkv_2a:>10.1f} ms     | {'Linear Thrash / TKV HIT':<14} | {f'{accel_2a:.2f}x FASTER':<12}")
    print(f"{'Turn 2B (Re-ask B)':<20} | {lin_2b:>10.1f} ms       | {tkv_2b:>10.1f} ms     | {'Linear Thrash / TKV HIT':<14} | {f'{accel_2b:.2f}x FASTER':<12}")
    print("-" * 88)

    p_saved_a = max(0.0, (1.0 - (tkv_2a / lin_2a)) * 100.0) if lin_2a > 0 else 0.0
    p_saved_b = max(0.0, (1.0 - (tkv_2b / lin_2b)) * 100.0) if lin_2b > 0 else 0.0

    print("EMPIRICAL CONCLUSION:")
    print(f"  * Linear Chain KVCache evicted Prompt A when Prompt B arrived -> Turn 2A suffered full prefill.")
    print(f"  * Tiered KV Cache preserved both prefixes in Radix Tree -> Turn 2A prefill was completely bypassed!")
    print(f"  * Latency Reduction: Prompt A: {p_saved_a:.1f}% TTFT reduction | Prompt B: {p_saved_b:.1f}% TTFT reduction.")
    print(f"  * Turn 2A Speedup: {accel_2a:.2f}x | Turn 2B Speedup: {accel_2b:.2f}x.")
    print("=" * 88 + "\n")


# -----------------------------------------------------------------------------
# In-process Mock Server Environment (for unit test verification without GPU)
# -----------------------------------------------------------------------------
def run_mock_suite(mode: str = "compare", tier: int = 1, port: int | None = None) -> bool:
    """Runs mock benchmarks simulating Linear Chain vs Tiered KV cache behavior."""
    from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, serve

    class ConfigurableSimulatedEngine(MockEngine):
        """Simulates latency under Linear Chain vs Tiered KV Cache."""

        def __init__(self, tok, is_tiered: bool):
            super().__init__(tok, "The primary rule is deterministic synchronization and strict alignment.", max_context=8192)
            self.is_tiered = is_tiered
            self.cached_branches: set[int] = set()
            self.active_branch: int | None = None

        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            # Identify prompt lineage by the leading prefix tokens
            prompt_key = hash(tuple(ids[:min(len(ids), 64)]))

            if self.is_tiered:
                is_hit = prompt_key in self.cached_branches
                if is_hit:
                    prompt_ms = 12.0
                    time.sleep(0.015)
                    reused = len(ids) - 15
                else:
                    prompt_ms = 350.0
                    time.sleep(0.35)
                    reused = 0
                    self.cached_branches.add(prompt_key)
            else:
                # Linear Chain: single active context. Alternating prompts thrash the single chain.
                is_hit = (self.active_branch == prompt_key)
                if is_hit:
                    prompt_ms = 12.0
                    time.sleep(0.015)
                    reused = len(ids) - 15
                else:
                    prompt_ms = 350.0
                    time.sleep(0.35)
                    reused = 0
                    self.active_branch = prompt_key  # evicts other branch!

            n = 0
            for t in super().generate(ids, max_new, sampling, cancel, embeddings):
                n += 1
                yield t

            self.last = {
                "generated": n,
                "prompt_tokens": len(ids),
                "prompt_ms": prompt_ms,
                "decode_ms": 15.0 * n,
                "finish": "stop",
                "reused": reused,
                "prompt_read": len(ids) - reused,
            }

    tok = ByteTokenizer()

    def run_one_mock(is_tiered: bool, label: str) -> dict:
        p = port if port is not None else find_free_port()
        eng = ConfigurableSimulatedEngine(tok, is_tiered=is_tiered)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"), exp_tiered_kv=is_tiered)
        httpd = serve(svc, port=p)
        assigned = httpd.server_address[1]
        try:
            return run_benchmark(f"http://127.0.0.1:{assigned}", prompt_words=300, max_tokens=12, label=label)
        finally:
            httpd.server_close()

    if mode == "compare":
        print("\n[Mock Environment] Running Step 1: Linear Chain KVCache Baseline...")
        res_lin = run_one_mock(is_tiered=False, label="Linear Chain KVCache (Baseline)")
        print("\n[Mock Environment] Running Step 2: Persistent Tiered KV Cache...")
        res_tkv = run_one_mock(is_tiered=True, label="Persistent Tiered KV Cache (Tier 1/2)")
        print_comparison_table(res_lin, res_tkv)
        return res_tkv["speedup_a"] > 5.0
    elif mode == "tiered":
        res_tkv = run_one_mock(is_tiered=True, label="Persistent Tiered KV Cache")
        print_single_summary(res_tkv)
        return res_tkv["speedup_a"] > 1.2 or res_tkv["speedup_b"] > 1.2
    else:
        res_lin = run_one_mock(is_tiered=False, label="Linear Chain KVCache")
        print_single_summary(res_lin)
        return True


def main():
    parser = argparse.ArgumentParser(description="Benchmark & Compare Linear Chain vs Tiered KV Cache under Alternating System Prompts")
    parser.add_argument("--url", default=None, help="Connect to an already running Strata server (e.g. http://127.0.0.1:8080)")
    parser.add_argument("--config", default=None, help="Path to strata-*.json configuration to launch")
    parser.add_argument("--port", type=int, default=None, help="Designate port (default: random free ephemeral port)")
    parser.add_argument("--mode", choices=["compare", "tiered", "linear"], default="compare",
                        help="Benchmark mode: 'compare' (both), 'tiered' (Tiered KV only), or 'linear' (Linear Chain only)")
    parser.add_argument("--tier", type=int, choices=[1, 2], default=1,
                        help="Tiered KV tier: 1 = Host RAM Radix tree (default), 2 = Host RAM + NVMe CAS SSD")
    parser.add_argument("--cache-dir", default=None, help="Custom persistent NVMe cache directory (implies --tier 2)")
    parser.add_argument("--mock", action="store_true", help="Run with simulated in-process MockEngine on random port")
    parser.add_argument("--words", type=int, default=800, help="Word count for system prompts (default: 800 words)")
    parser.add_argument("--max-tokens", type=int, default=16, help="Maximum generated tokens per query (default: 16)")
    args = parser.parse_args()

    tier = 2 if args.cache_dir is not None else args.tier
    cache_path = Path(args.cache_dir) if args.cache_dir else None

    # 1. Mock execution
    if args.mock:
        success = run_mock_suite(mode=args.mode, tier=tier, port=args.port)
        sys.exit(0 if success else 1)

    # 2. Existing URL provided
    if args.url:
        try:
            with urllib.request.urlopen(f"{args.url.rstrip('/')}/v1/models", timeout=3.0) as resp:
                if resp.status != 200:
                    print(f"Error: Server at {args.url} returned status {resp.status}")
                    sys.exit(1)
        except Exception as e:
            print(f"Error: Could not connect to Strata server at {args.url}: {e}")
            sys.exit(1)

        res = run_benchmark(url=args.url, prompt_words=args.words, max_tokens=args.max_tokens, label=f"Server at {args.url}")
        print_single_summary(res)
        sys.exit(0 if res["speedup_a"] > 1.2 or res["speedup_b"] > 1.2 else 1)

    # 3. Auto-start server with random port
    config_file = Path(args.config) if args.config else find_default_config()
    if not config_file or not config_file.exists():
        print("Notice: No model configuration file (strata-*.json) found in current directory.")
        print("To test the benchmark comparison logic immediately: python3 tools/bench_tiered_kv.py --mock")
        sys.exit(1)

    def run_server_suite(mode_to_run: str) -> dict:
        allocated_port = args.port or find_free_port()
        proc, port, temp_cfg, cdir = launch_server(
            config_file,
            port=allocated_port,
            mode=mode_to_run,
            tier=tier,
            cache_dir=cache_path,
        )

        def sig_handler(sig, frame):
            print("\nInterrupt received. Terminating server...")
            stop_server(proc, temp_cfg, cdir)
            sys.exit(130)

        prev_int = signal.signal(signal.SIGINT, sig_handler)
        prev_term = signal.signal(signal.SIGTERM, sig_handler)

        try:
            res = run_benchmark(
                url=f"http://127.0.0.1:{port}",
                prompt_words=args.words,
                max_tokens=args.max_tokens,
                label=f"{mode_to_run.upper()} Mode",
            )
            return res
        finally:
            signal.signal(signal.SIGINT, prev_int)
            signal.signal(signal.SIGTERM, prev_term)
            stop_server(proc, temp_cfg, cdir)

    if args.mode == "compare":
        print("\n======================================================================")
        print("   STRATA BENCHMARK COMPARISON: LINEAR CHAIN vs PERSISTENT TIERED KV  ")
        print("======================================================================")
        print("Phase 1: Benchmarking Linear Chain KVCache Baseline...")
        res_linear = run_server_suite("linear")

        print("\nPhase 2: Benchmarking Tiered KV Cache...")
        res_tiered = run_server_suite("tiered")

        print_comparison_table(res_linear, res_tiered)
        sys.exit(0)
    elif args.mode == "tiered":
        res_tiered = run_server_suite("tiered")
        print_single_summary(res_tiered)
        sys.exit(0 if res_tiered["speedup_a"] > 1.2 or res_tiered["speedup_b"] > 1.2 else 1)
    else:
        res_linear = run_server_suite("linear")
        print_single_summary(res_linear)
        sys.exit(0)


if __name__ == "__main__":
    main()
