// tests/core/ring_wait_test.cpp - RingWait's decision table (ring_wait.hpp): a quiet wait fires at the limit,
// a wait whose file tier is still busy holds on (once noted) up to the allowance and until 10 s after the last
// busy sample, the allowance's end fires even while busy, and +16 over the last sample is busy while +14 is not.
// Synthetic clocks and an injected activity counter: no disk, no GPU, no sleeping.  The env knobs are read into
// function-local statics on first use, so the test fixes them (1 s limit, 15 s allowance: the 10 s quiet rule
// must fit inside the allowance) before the first wait.
#include "strata/core/ring_wait.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>

namespace {
int g_fail = 0;
void check(bool ok, const char* what) {
    std::printf("  %-90s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}

uint64_t g_activity = 0;
uint64_t fake_activity() { return g_activity; }

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv((std::string(name) + "=" + value).c_str());
#else
    setenv(name, value, 1);
#endif
}
}  // namespace

int main() {
    set_env("STRATA_RING_TIMEOUT_S", "1");
    set_env("STRATA_RING_IO_S", "15");

    using strata::core::RingWait;

    {   // a quiet wait: nothing fires before the limit, and the limit itself fires (as the old 20 s did)
        g_activity = 0;
        RingWait w(1000, fake_activity);
        check(!w.should_fire(1500), "before the limit the wait holds");
        check(!w.noted, "before the limit nothing is noted");
        check(w.should_fire(2000), "a quiet wait fires at the limit");
        check(!w.noted, "a quiet wait is never noted");
    }
    {   // a busy file tier: hold past the limit, note once, fire 10 s after the last busy sample
        g_activity = 100;
        RingWait w(1000, fake_activity);
        g_activity = 500;
        check(!w.should_fire(3000), "a busy file tier holds the wait past the limit");
        check(w.noted, "the first hold notes the slow storage");
        check(!w.should_fire(3100), "still within 10 s of the last busy sample: hold");
        g_activity = 500;
        check(!w.should_fire(12999), "9.999 s after the last busy sample: still hold");
        check(w.should_fire(13000), "10 s after the last busy sample: fire");
        check(w.noted, "the note stays noted");
    }
    {   // the busy threshold is +16 over the last sample, as the #29 watchdog counts it
        g_activity = 100;
        RingWait w(1000, fake_activity);
        g_activity = 500;
        check(!w.should_fire(3000), "wait: a busy sample opens the hold");
        g_activity = 514;
        check(!w.should_fire(4000), "wait: +14 over the last sample is not busy");
        g_activity = 531;
        check(!w.should_fire(5000), "wait: +17 over the last sample is busy and holds");
        check(w.noted, "the busy sample notes");
        g_activity = 531;
        check(w.should_fire(15000), "10 s after that last busy sample: fire");
    }
    {   // the allowance's end fires even while the file tier is still busy
        g_activity = 10;
        RingWait w(1000, fake_activity);
        g_activity = 30;
        check(!w.should_fire(2999), "busy inside the allowance: hold");
        g_activity = 50;
        check(!w.should_fire(15999), "busy at the allowance's last moment: hold");
        g_activity = 70;
        check(w.should_fire(16000), "the allowance's end fires while busy");
    }
    {   // reset: the same object waits again from the new start
        g_activity = 1000;
        RingWait w(1000, fake_activity);
        check(w.should_fire(2000), "the first wait fires");
        w.reset(100000, fake_activity);
        check(!w.should_fire(100500), "after reset the wait holds again");
        check(w.should_fire(101000), "after reset the limit is the new start's");
    }
    std::printf(g_fail ? "FAILED (%d)\n" : "ALL PASSED\n", g_fail);
    return g_fail ? 1 : 0;
}
