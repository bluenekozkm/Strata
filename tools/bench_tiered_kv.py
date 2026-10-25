#!/usr/bin/env python3
"""Benchmark Persistent Tiered KV Cache under alternating multi-system-prompt workloads.

Scenario:
  Two divergent system prompts (System A: Code Architect, System B: Research Analyst).
  - Round 1 (Cold): No KV cache exists for either prompt. Both A and B must be fully
    prefilled from scratch. When switching from A to B, A is evicted from active VRAM
    to Tier 1 (Host Pinned RAM Radix Tree) / Tier 2 (NVMe CAS storage).
  - Round 2 (Warm): Switching back to A, then B. The Tiered KV Cache identifies the
    cached prefix in the Radix forest and DMA transfers the resident blocks, bypassing
    prefill computation.
  - Verification: If Round 2 achieves significantly lower TTFT / prefill time (cache hit),
    the Tiered KV Cache system is confirmed working.

Server Lifecycle:
  - Automatically allocates a free random port to avoid port collision with any running processes.
  - Automatically starts the server with --kv-multi-system-prompt and --kv-disk-cache enabled.
  - Cleanly shuts down the server upon benchmark completion or interrupt (flushing manifest).

Usage:
  # Automatic server launch on a random port with auto-detected config (strata-*.json):
  python3 tools/bench_tiered_kv.py

  # With specific configuration file:
  python3 tools/bench_tiered_kv.py --config strata-gyro-gyro-m.json

  # Connect to an already running server:
  python3 tools/bench_tiered_kv.py --url http://127.0.0.1:8080

  # Standalone mock verification (no GPU / model required):
  python3 tools/bench_tiered_kv.py --mock
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
    """Executes a streaming request against /v1/chat/completions to accurately measure TTFT and prefill."""
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
    cache_dir: Path | None = None,
) -> tuple[subprocess.Popen, int, Path, Path]:
    """Prepares and launches the resident server with Tiered KV cache enabled on a designated or random port."""
    if port is None:
        port = find_free_port()

    if cache_dir is None:
        cache_dir = Path(tempfile.mkdtemp(prefix=f"strata_tiered_kv_{port}_"))
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    args = list(cfg.get("args", []))

    def set_arg(flag: str, val: str | None = None):
        if flag not in args:
            args.append(flag)
            if val is not None:
                args.append(str(val))
        elif val is not None:
            idx = args.index(flag)
            if idx + 1 < len(args):
                args[idx + 1] = str(val)

    # Ensure multi-branch radix tree, tiered NVMe disk cache and host conversation pool are enabled
    set_arg("--kv-multi-system-prompt")
    set_arg("--kv-disk-cache", str(cache_dir))
    set_arg("--kv-disk-budget", "10240")
    set_arg("--conversation-cache-mib", "4096")
    set_arg("--conversation-cache-slots", "8")
    set_arg("--prompt-cache-root", "256")
    set_arg("--prompt-cache", "8")

    cfg["args"] = args
    cfg["port"] = port
    cfg["log"] = str(cache_dir / f"strata_server_{port}.log")

    temp_cfg_path = cache_dir / f"strata_bench_config_{port}.json"
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

    print(f"[Server Auto-Start] Launching REAL resident Strata engine on random port: {port}")
    print(f"  Engine Binary: {cfg.get('exe', 'engine/strata')}")
    print(f"  Model GGUF:    {Path(cfg.get('args', [''])[3]).name if len(cfg.get('args', [])) > 3 else 'auto'}")
    print(f"  Config:        {config_path.name}")
    print(f"  KV Cache Dir:  {cache_dir}")
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
                    sys.stdout.write(f"  [strata-engine] {line}")
                    sys.stdout.flush()
        except Exception:
            pass

    log_thread = threading.Thread(target=stream_logs, daemon=True)
    log_thread.start()

    try:
        wait_ready(port, proc)
        print(f"[Server Ready] Server successfully loaded and listening on 127.0.0.1:{port}\n")
    except Exception:
        stop_server(proc, temp_cfg_path, cache_dir)
        raise

    return proc, port, temp_cfg_path, cache_dir


def stop_server(proc: subprocess.Popen, temp_cfg: Path | None, cache_dir: Path | None) -> None:
    """Gracefully terminates the server to flush the Tiered KV manifest and cleans up temp files."""
    if proc.poll() is None:
        print("\n[Server Shutdown] Sending SIGINT to flush Tiered KV manifest atomically...")
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


def run_benchmark(url: str, prompt_words: int = 800, max_tokens: int = 16) -> dict:
    """Runs the 2-System-Prompt alternating benchmark."""
    print("======================================================================")
    print("      STRATA PERSISTENT TIERED KV CACHE - EMPIRICAL BENCHMARK         ")
    print("======================================================================")
    print(f"Target URL:         {url}")
    print(f"System Prompt Size: ~{prompt_words} words per prompt (~1,200 tokens)")
    print(f"Sampling:           max_tokens={max_tokens}, temperature=0 (greedy)")
    print("----------------------------------------------------------------------\n")

    sys_a = generate_system_prompt("code", target_words=prompt_words)
    sys_b = generate_system_prompt("compliance", target_words=prompt_words)

    initial_metrics = fetch_metrics(url).get("tiered_kv", {})
    if initial_metrics:
        print(f"Initial Tiered KV Telemetry: {initial_metrics}\n")

    results = []

    # -------------------------------------------------------------------------
    # ROUND 1: Cold Start (No KV cache -> Full prefill for both prompts)
    # -------------------------------------------------------------------------
    print("[Phase 1: Cold Prefill / Cache Miss Baseline]")

    # 1. Turn 1A: System Prompt A
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
    results.append(("Round 1 (Cold)", "Prompt A", res_1a))

    # 2. Turn 1B: System Prompt B (Switch context, evicting A to Tier 1 / Tier 2)
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
    results.append(("Round 1 (Cold)", "Prompt B", res_1b))

    # -------------------------------------------------------------------------
    # ROUND 2: Warm Retrieval (Differential switch -> Cache hit expected)
    # -------------------------------------------------------------------------
    print("\n[Phase 2: Warm Multi-Branch Radix Retrieval (Tiered KV Cache)]")

    # 3. Turn 2A: System Prompt A (Switch back to A -> Tiered KV Cache hit)
    print("  -> Turn 2A: Switching back to System Prompt A (Cache Hit Expected)...")
    res_2a = post_stream(
        url,
        messages=[
            {"role": "system", "content": sys_a},
            {"role": "user", "content": "What is rule #2? Answer in 1 sentence."},
        ],
        max_tokens=max_tokens,
    )
    print(f"     TTFT: {res_2a['ttft_ms']:.1f} ms | Total: {res_2a['total_ms']:.1f} ms | Timings: {res_2a.get('timings') or 'N/A'}")
    results.append(("Round 2 (Warm)", "Prompt A", res_2a))

    # 4. Turn 2B: System Prompt B (Switch back to B -> Tiered KV Cache hit)
    print("  -> Turn 2B: Switching back to System Prompt B (Cache Hit Expected)...")
    res_2b = post_stream(
        url,
        messages=[
            {"role": "system", "content": sys_b},
            {"role": "user", "content": "What is Section 2 constraint? Answer in 1 sentence."},
        ],
        max_tokens=max_tokens,
    )
    print(f"     TTFT: {res_2b['ttft_ms']:.1f} ms | Total: {res_2b['total_ms']:.1f} ms | Timings: {res_2b.get('timings') or 'N/A'}")
    results.append(("Round 2 (Warm)", "Prompt B", res_2b))

    final_metrics = fetch_metrics(url).get("tiered_kv", {})

    # -------------------------------------------------------------------------
    # Analysis & Verification
    # -------------------------------------------------------------------------
    print("\n======================================================================")
    print("                     BENCHMARK RESULTS SUMMARY                        ")
    print("======================================================================")
    print(f"{'Phase':<16} | {'Prompt':<10} | {'TTFT (ms)':<10} | {'Prefill Time':<14} | {'Speedup':<10}")
    print("-" * 70)

    # Compute Speedup for A and B
    speedup_a = (res_1a["ttft_ms"] / res_2a["ttft_ms"]) if res_2a["ttft_ms"] > 0 else 1.0
    speedup_b = (res_1b["ttft_ms"] / res_2b["ttft_ms"]) if res_2b["ttft_ms"] > 0 else 1.0

    p_ms_1a = res_1a.get("timings", {}).get("prompt_ms", "N/A")
    p_ms_1b = res_1b.get("timings", {}).get("prompt_ms", "N/A")
    p_ms_2a = res_2a.get("timings", {}).get("prompt_ms", "N/A")
    p_ms_2b = res_2b.get("timings", {}).get("prompt_ms", "N/A")

    print(f"{'Round 1 (Cold)':<16} | {'Prompt A':<10} | {res_1a['ttft_ms']:<10.1f} | {str(p_ms_1a) + ' ms':<14} | {'1.0x (baseline)':<10}")
    print(f"{'Round 1 (Cold)':<16} | {'Prompt B':<10} | {res_1b['ttft_ms']:<10.1f} | {str(p_ms_1b) + ' ms':<14} | {'1.0x (baseline)':<10}")
    print(f"{'Round 2 (Warm)':<16} | {'Prompt A':<10} | {res_2a['ttft_ms']:<10.1f} | {str(p_ms_2a) + ' ms':<14} | {f'{speedup_a:.2f}x':<10}")
    print(f"{'Round 2 (Warm)':<16} | {'Prompt B':<10} | {res_2b['ttft_ms']:<10.1f} | {str(p_ms_2b) + ' ms':<14} | {f'{speedup_b:.2f}x':<10}")
    print("-" * 70)

    if final_metrics:
        print(f"Final Telemetry (/metrics):")
        print(f"  * System Prompt Branches: {final_metrics.get('system_prompt_branches', 0)}")
        print(f"  * System Prompt Switches: {final_metrics.get('system_prompt_switches', 0)}")
        print(f"  * System Prompt Hits:     {final_metrics.get('system_prompt_hits', 0)}")

    passed = (speedup_a > 1.2 or speedup_b > 1.2) or (
        final_metrics.get("system_prompt_switches", 0) >= 2 and final_metrics.get("system_prompt_hits", 0) >= 1
    )

    print("\n----------------------------------------------------------------------")
    if passed:
        print("VERDICT: [PASS] Persistent Tiered KV Cache is ACTIVE and FUNCTIONING.")
        print(f"         Prompt A accelerated by {speedup_a:.2f}x | Prompt B accelerated by {speedup_b:.2f}x.")
        print("         Multi-branch prefix retention and fast prefill bypass verified.")
    else:
        print("VERDICT: [INCONCLUSIVE / FAIL] Cache acceleration was not observed.")
        print("         Ensure the server was started with --kv-multi-system-prompt and --kv-disk-cache.")
    print("----------------------------------------------------------------------\n")

    return {
        "passed": passed,
        "speedup_a": speedup_a,
        "speedup_b": speedup_b,
        "results": results,
        "final_metrics": final_metrics,
    }


# -----------------------------------------------------------------------------
# Standalone Mock Server Environment (for unit test verification without GPU)
# -----------------------------------------------------------------------------
def run_mock_benchmark(port: int | None = None):
    """Runs the benchmark against an in-process MockEngine on a random port."""
    from serve.server import ByteTokenizer, ChatTemplate, MockEngine, Service, serve

    class TieredKvSimulatedEngine(MockEngine):
        """Simulates cold prefill latency (350 ms) and warm Tiered KV cache hits (12 ms)."""

        def __init__(self, tok):
            super().__init__(tok, "The primary rule is deterministic synchronization and strict alignment.", max_context=8192)
            self.cached_branches: set[int] = set()

        def generate(self, ids, max_new, sampling, cancel, embeddings=None):
            tkv = sampling.get("tiered_kv") or {}
            branch_id = tkv.get("branch_id", 0)
            is_hit = branch_id in self.cached_branches

            if is_hit:
                prompt_ms = 12.0
                time.sleep(0.015)
                reused = len(ids) - 15
            else:
                prompt_ms = 350.0
                time.sleep(0.35)
                reused = 0
                self.cached_branches.add(branch_id)

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

    bind_port = port if port is not None else find_free_port()
    print(f"[Mock Mode] Spinning up in-process Strata Server on random port {bind_port}...")
    tok = ByteTokenizer()
    engine = TieredKvSimulatedEngine(tok)
    svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
    httpd = serve(svc, port=bind_port)
    assigned_port = httpd.server_address[1]
    url = f"http://127.0.0.1:{assigned_port}"

    try:
        return run_benchmark(url=url, prompt_words=300, max_tokens=12)
    finally:
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Persistent Tiered KV Cache Alternating System Prompt Benchmark")
    parser.add_argument("--url", default=None, help="Connect to an already running Strata server (e.g. http://127.0.0.1:8080)")
    parser.add_argument("--config", default=None, help="Path to strata-*.json configuration to launch")
    parser.add_argument("--port", type=int, default=None, help="Designate port (default: random free ephemeral port)")
    parser.add_argument("--mock", action="store_true", help="Run with simulated in-process MockEngine on random port")
    parser.add_argument("--words", type=int, default=800, help="Word count for system prompts (default: 800 words)")
    parser.add_argument("--max-tokens", type=int, default=16, help="Maximum generated tokens per query (default: 16)")
    args = parser.parse_args()

    # 1. Mock execution
    if args.mock:
        res = run_mock_benchmark(port=args.port)
        sys.exit(0 if res["passed"] else 1)

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

        res = run_benchmark(url=args.url, prompt_words=args.words, max_tokens=args.max_tokens)
        sys.exit(0 if res["passed"] else 1)

    # 3. Auto-start server with random port
    config_file = Path(args.config) if args.config else find_default_config()
    if not config_file or not config_file.exists():
        print("Notice: No model configuration file (strata-*.json) found in current directory.")
        print("To test the benchmark logic immediately: python3 tools/bench_tiered_kv.py --mock")
        sys.exit(1)

    allocated_port = args.port or find_free_port()
    proc, port, temp_cfg, cache_dir = launch_server(config_file, port=allocated_port)

    # Setup termination signal handler to ensure server is killed if user presses Ctrl+C
    def sig_handler(sig, frame):
        print("\nInterrupt received. Terminating server...")
        stop_server(proc, temp_cfg, cache_dir)
        sys.exit(130)

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    try:
        res = run_benchmark(url=f"http://127.0.0.1:{port}", prompt_words=args.words, max_tokens=args.max_tokens)
        sys.exit(0 if res["passed"] else 1)
    finally:
        stop_server(proc, temp_cfg, cache_dir)


if __name__ == "__main__":
    main()
