"""The engine's OOM score: read it, ask for a value, and the one-time advice.

    python -m unittest serve.test_oom_guard -v
"""
import os
import unittest

from serve import server


class OomScoreTests(unittest.TestCase):
    def setUp(self):
        server._OOM_ADVICE_SHOWN = False

    def test_own_score_is_readable_or_absent(self):
        adj = server.oom_score_adj(os.getpid())
        self.assertTrue(adj is None or -1000 <= adj <= 1000)

    def test_missing_process_is_absent(self):
        self.assertIsNone(server.oom_score_adj(999_999_999))
        self.assertIn("No such", server.set_oom_score_adj(999_999_999, 0))

    def test_asking_for_the_current_value_is_accepted(self):
        cur = server.oom_score_adj(os.getpid())
        if cur is None:
            self.skipTest("no /proc on this platform")
        self.assertEqual(server.set_oom_score_adj(os.getpid(), cur), "")   # no change: always allowed

    def test_advice_only_for_a_high_score(self):
        cur = server.oom_score_adj(os.getpid())
        if cur is None:
            self.skipTest("no /proc on this platform")
        if cur < 100:
            self.assertEqual(server.oom_advice(os.getpid()), "")
            self.assertFalse(server._OOM_ADVICE_SHOWN)      # nothing was shown, so another engine may still show it
        else:
            self.assertIn("oom_score_adj", server.oom_advice(os.getpid()))
            self.assertEqual(server.oom_advice(os.getpid()), "")   # once per server run


if __name__ == "__main__":
    unittest.main()
