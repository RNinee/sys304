import unittest

from trigger import should_retrain


class TriggerTest(unittest.TestCase):
    def test_time_threshold_fires_when_never_run(self):
        decision = should_retrain(
            now=1_000.0,
            last_run=None,
            interval_seconds=60,
            mean_confidence=0.9,
            confidence_floor=0.55,
        )
        self.assertTrue(decision.run)
        self.assertEqual(decision.reason, "time")

    def test_time_threshold_fires_after_interval(self):
        decision = should_retrain(
            now=200.0,
            last_run=100.0,
            interval_seconds=50,
            mean_confidence=0.9,
            confidence_floor=0.55,
        )
        self.assertTrue(decision.run)
        self.assertIn("time", decision.reason)

    def test_confidence_drop_fires_before_the_interval(self):
        decision = should_retrain(
            now=110.0,
            last_run=100.0,
            interval_seconds=3_600,
            mean_confidence=0.42,
            confidence_floor=0.55,
        )
        self.assertTrue(decision.run)
        self.assertEqual(decision.reason, "confidence")

    def test_either_condition_is_enough(self):
        decision = should_retrain(
            now=5_000.0,
            last_run=0.0,
            interval_seconds=60,
            mean_confidence=0.2,
            confidence_floor=0.55,
        )
        self.assertTrue(decision.run)
        self.assertEqual(decision.reason, "time+confidence")

    def test_idle_when_recent_and_confident(self):
        decision = should_retrain(
            now=120.0,
            last_run=100.0,
            interval_seconds=3_600,
            mean_confidence=0.88,
            confidence_floor=0.55,
        )
        self.assertFalse(decision.run)
        self.assertEqual(decision.reason, "idle")

    def test_force_overrides_idle(self):
        decision = should_retrain(
            now=120.0,
            last_run=100.0,
            interval_seconds=3_600,
            mean_confidence=0.88,
            confidence_floor=0.55,
            force=True,
        )
        self.assertTrue(decision.run)
        self.assertEqual(decision.reason, "forced")


if __name__ == "__main__":
    unittest.main()
