"""The v1 telemetry legacy adapter: an older single-slot server's /metrics + /v1/status shaped as the
versioned schema-1 contract.  The server answers /v1/telemetry with it, so the shape and every null
here are part of the API surface — do not "improve" them without bumping the schema.

WHY IT LIVES IN SERVE.  It was the Terminal monitor's read-side normalizer (strata_cli/client.py) and the
server reused it for its own endpoint; with the Terminal gone from the tree the server keeps the only
copy.  It is not updated from upstream's client.py anymore.
"""
from __future__ import annotations

import math
from typing import Any


def complete_slots(work: dict) -> list[dict] | None:
    """Return the effective slot set only when its identity and states are sound."""
    slots, effective = work.get("slots"), work.get("slots_effective")
    if (not isinstance(slots, list) or not isinstance(effective, int) or
            isinstance(effective, bool) or effective <= 0 or len(slots) != effective):
        return None
    identities: set[int] = set()
    for row in slots:
        if not isinstance(row, dict):
            return None
        index, state = row.get("slot"), row.get("state")
        if (not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < effective or
                index in identities or state not in ("idle", "reading", "decoding")):
            return None
        identities.add(index)
    return slots


def _number(value: Any) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return value


def _short(value: Any, limit: int = 120) -> str | None:
    if not isinstance(value, str):
        return None
    # Data is untrusted even if it came from the user's own model/server.
    return "".join(c if c.isprintable() and c not in "\x1b\x7f" else " " for c in value)[:limit]


def normalize_legacy(metrics: dict[str, Any], status: dict[str, Any]) -> dict[str, Any]:
    """Keep null distinct from zero; never infer group/stage work from GPU count."""
    live = metrics.get("live") if isinstance(metrics.get("live"), dict) else {}
    engine = metrics.get("engine") if isinstance(metrics.get("engine"), dict) else {}
    hardware = metrics.get("hardware") if isinstance(metrics.get("hardware"), dict) else {}
    static = metrics.get("hardware_static") if isinstance(metrics.get("hardware_static"), dict) else {}
    concurrency = status.get("concurrency") if isinstance(status.get("concurrency"), dict) else {}
    slots = live.get("slots") if isinstance(live.get("slots"), list) else []
    devices = hardware.get("gpus") if isinstance(hardware.get("gpus"), list) else []
    safe_slots = []
    for row in slots[:64]:
        if isinstance(row, dict):
            safe_slots.append({k: row.get(k) for k in ("slot", "group", "replica", "request_id", "state", "held_tokens",
                                                       "prompt_tokens", "prompt_read", "generated", "elapsed_s", "tok_s")})
    effective_slots = _number(concurrency.get("serving"))
    reported_slots = complete_slots({"slots": safe_slots, "slots_effective": effective_slots})
    safe_devices = []
    for row in devices[:16]:
        if isinstance(row, dict):
            safe_devices.append({"index": row.get("index"), "name": _short(row.get("name")),
                                 "mem_used": _number(row.get("mem_used")),
                                 "mem_total": _number(row.get("mem_total")),
                                 "util": _number(row.get("util")), "temp": _number(row.get("temp")),
                                 "power": _number(row.get("power"))})
    # Old single-GPU servers report only aggregate GPU fields. It is one device only
    # when the static sampler says so; never duplicate an aggregate into four rows.
    if not safe_devices and static.get("gpu_count") == 1 and static.get("gpu_name"):
        safe_devices = [{"index": 0, "name": _short(static.get("gpu_name")),
                         "mem_used": _number(hardware.get("gpu_mem_used")),
                         "mem_total": _number(hardware.get("gpu_mem_total")),
                         "util": _number(hardware.get("gpu_util")), "temp": _number(hardware.get("gpu_temp")),
                         "power": _number(hardware.get("gpu_power"))}]
    recent = []
    for row in (metrics.get("requests") if isinstance(metrics.get("requests"), list) else [])[:50]:
        if isinstance(row, dict):
            recent.append({k: row.get(k) for k in ("id", "time", "duration_s", "finish", "prompt_tokens",
                                                      "reused", "output_tokens", "prompt_ms", "decode_ms",
                                                      "first_engine_token_s")})
    return {"schema_version": 1, "source": "legacy-adapter", "sampled_at": _number(metrics.get("time")),
            "instance": {"boot_time": _number(status.get("started")), "server": _short(status.get("service"))},
            "model": {"id": _short(status.get("model")), "loaded": status.get("loaded") if isinstance(status.get("loaded"), bool) else None,
                      "context": _number(status.get("cache_max_tokens"))},
            "work": {"slots_requested": None, "slots_effective": effective_slots,
                     "active": _number(live.get("running")) if "running" in live else (1 if live.get("state") in ("reading", "generating") else 0),
                     "occupied_slots": (sum(row["state"] != "idle" for row in reported_slots)
                                        if reported_slots is not None else None),
                     "queued": _number(live.get("queued")), "waiting": _number(live.get("waiting")),
                     "outside_slots": _number(live.get("outside_slots")), "slots": safe_slots,
                     "groups": _number(engine.get("batch_groups")), "effective_rows_window": None,
                     "batch_active_rows_window": None, "batch_sample_windows": None,
                     "batch_sample_rows": None, "batch_sample_active_rows": None,
                     "batch_sample_span_s": None, "batch_sample_age_s": None,
                     "stage_in_flight": [], "stage_sample_span_s": None, "stage_sample_age_s": None,
                     "requests": [], "requests_omitted": None, "max_queue": None,
                     "admission_limit": None, "admission_reserved": None,
                     "admission_rejected": None, "oldest_queue_s": None,
                     "queue_timeout_s": None, "queue_timeouts": None,
                     "client_disconnect_cancels": None},
            "rates": {"answer_tok_s": _number(live.get("tok_s")),
                      "answer_window_s": _number(live.get("tok_s_window_s")),
                      "prefill_tok_s": _number(live.get("prefill_tok_s_mean"))},
            "topology": {"mode": None, "stages": []}, "devices": safe_devices,
            "ram": {"used": _number(hardware.get("ram_used")), "total": _number(hardware.get("ram_total"))},
            "recent": recent, "totals": {k: _number(v) for k, v in
                                          (metrics.get("totals") if isinstance(metrics.get("totals"), dict) else {}).items()
                                          if k in ("since", "requests", "prompt_tokens", "output_tokens")},
            "quality": {"request_correlation": "unavailable", "batch_window_rows": "unavailable",
                        "batch_window_source": None,
                        "stage_in_flight": "unavailable", "stage_in_flight_source": None,
                        "per_device": "measured" if safe_devices else "unavailable"}}
