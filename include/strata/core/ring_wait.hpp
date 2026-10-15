// include/strata/core/ring_wait.hpp - the layer-doorbell waits' limit, and its slow-storage allowance.
//
// The verify window and the plain token path wait, per layer, for the GPU's doorbell (the mapped sequence
// word).  A wait past the limit ends the engine (#267): a doorbell nobody will raise means the GPU spins for
// ever.  But the same lateness has a second cause that waiting longer does fix - the OS is still reading the
// file tier for us (a cold page cache, slow storage, a thrashing host).  Those waits follow the #29 watchdog's
// rule: they fire only once the file tier has been quiet for 10 s, and hold on for at most STRATA_RING_IO_S
// (default 10 x the limit) while it is being read.
#pragma once

#include "strata/core/progress.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif

namespace strata::core {

/// #1407: the file tier's activity as the OS counts it (Linux: major page faults and bytes read from storage).
/// A growing count says "the step is slow, not stuck".  Zero where unavailable.
inline uint64_t file_tier_activity() {
#if defined(__linux__)
    uint64_t total = 0;
    if (FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[1024] = {};
        const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
        std::fclose(f);
        const char* p = std::strrchr(buf, ')');
        if (p != nullptr && n > 0) {
            // after "(comm)": state ppid pgrp session tty tpgid flags minflt cminflt majflt
            unsigned long long majflt = 0;
            if (std::sscanf(p + 2, "%*c %*d %*d %*d %*d %*d %*u %*u %*u %llu", &majflt) == 1) total += majflt;
        }
    }
    if (FILE* f = std::fopen("/proc/self/io", "r")) {
        char line[256];
        while (std::fgets(line, sizeof(line), f) != nullptr) {
            unsigned long long v = 0;
            if (std::sscanf(line, "read_bytes: %llu", &v) == 1) total += v >> 20;   // MiB
        }
        std::fclose(f);
    }
    return total;
#elif defined(_WIN32)
    // #1984 #1971: Windows had no measure here, so a prompt waiting on mapped-file page-ins never got the I/O
    // allowance.  Bytes read (MiB) plus page faults / 8 (Windows counts soft faults too; a stuck engine makes
    // almost none).
    uint64_t total = 0;
    IO_COUNTERS io{};
    if (GetProcessIoCounters(GetCurrentProcess(), &io)) total += io.ReadTransferCount >> 20;
    PROCESS_MEMORY_COUNTERS pmc{};
    // >> 3, not >> 6: a host that pages hard reads from the pagefile at ~200 faults a second (#2049: 190/s, one
    // thread waiting on one page at a time), which at >> 6 is 3 a second and never counted as busy
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) total += pmc.PageFaultCount >> 3;
    return total;
#else
    return 0;
#endif
}

/// file_tier_activity, sampled at most once a second and shared by every wait in the process: a per-layer
/// wait that starts (and its once-a-second re-checks) then costs no syscall of its own.  Two threads may both
/// sample in the same second; the counts only grow, so either answer is fine.
inline uint64_t file_tier_activity_cached() {
    static std::atomic<int64_t> at_ms{0};
    static std::atomic<uint64_t> value{0};
    const int64_t now = progress_now_ms();
    if (now - at_ms.load(std::memory_order_relaxed) < 1000) return value.load(std::memory_order_relaxed);
    const uint64_t v = file_tier_activity();
    value.store(v, std::memory_order_relaxed);
    at_ms.store(now, std::memory_order_relaxed);
    return v;
}

/// How long a step waits for one layer's doorbell before it gives the window up (#267).  20 s by default;
/// STRATA_RING_TIMEOUT_S raises it (the name the SYCL engine already uses).
inline std::chrono::seconds ring_timeout() {
    static const long s = [] {
        const char* v = std::getenv("STRATA_RING_TIMEOUT_S");
        const long n = v ? std::atol(v) : 20;
        return n >= 1 ? n : 20L;
    }();
    return std::chrono::seconds(s);
}

/// The doorbell wait's slow-storage allowance: seconds a late doorbell may hold on while the file tier is
/// still being read (STRATA_RING_IO_S), default 10 x the doorbell limit; 0 turns the allowance off.
inline long ring_io_limit_s() {
    static const long s = [] {
        const char* v = std::getenv("STRATA_RING_IO_S");
        if (v) { const long n = std::atol(v); return n >= 0 ? n : 0L; }
        return ring_timeout().count() * 10L;
    }();
    return s;
}

/// One doorbell wait's decision, shared by every ring path.  Fires at the limit like before; past it, holds on
/// while the OS is still reading for us - up to the allowance, and only while the file tier was busy within
/// the last 10 s (seconds with no reads happen while the host thrashes, and still count as slow, #1407).
struct RingWait {
    /// Where the activity count comes from (tests replace it; the engine samples the shared once-a-second cache).
    using ActivityFn = uint64_t (*)();
    int64_t deadline_ms = 0;      ///< the limit, from the wait's start
    int64_t io_deadline_ms = 0;   ///< the allowance's end; 0 = no allowance: fire at the limit, as before
    int64_t last_sample_ms = 0;
    int64_t io_active_ms = 0;     ///< the last sample that saw the file tier busy (start - 10 s: quiet already)
    uint64_t io_last = 0;
    int64_t io_limit_s = 0;
    bool noted = false;           ///< the one stderr note about waiting on slow storage has gone out

    RingWait() = default;
    explicit RingWait(int64_t now_ms, ActivityFn activity = file_tier_activity_cached) { reset(now_ms, activity); }
    void reset(int64_t now_ms, ActivityFn activity = file_tier_activity_cached) {
        activity_ = activity;
        deadline_ms = now_ms + std::chrono::duration_cast<std::chrono::milliseconds>(ring_timeout()).count();
        io_limit_s = ring_io_limit_s();
        io_deadline_ms = io_limit_s > 0 ? now_ms + io_limit_s * 1000 : 0;
        last_sample_ms = now_ms;
        io_active_ms = now_ms - 10000;
        io_last = activity_();
        noted = false;
    }

    /// True when the wait should give the window up.  Cheap while early; past the limit it samples the file
    /// tier at most once a second.
    bool should_fire(int64_t now_ms) {
        if (now_ms < deadline_ms) return false;
        if (io_deadline_ms != 0 && now_ms < io_deadline_ms) {
            if (now_ms - last_sample_ms >= 1000) {
                const uint64_t a = activity_();
                if (a >= io_last + 16) io_active_ms = now_ms;   // 16 = faults + MiB read in a second: busy (#29)
                io_last = a;
                last_sample_ms = now_ms;
            }
            if (now_ms - io_active_ms < 10000) {
                if (!noted) {
                    noted = true;
                    std::fprintf(stderr, "strata serve: a layer's doorbell is late, but the file tier is still "
                                         "being read (slow storage or low RAM): waiting up to %lld s in all "
                                         "(STRATA_RING_IO_S)\n",
                                 (long long) io_limit_s);
                    std::fflush(stderr);
                }
                return false;
            }
        }
        return true;
    }

  private:
    ActivityFn activity_ = file_tier_activity_cached;
};

}  // namespace strata::core
