import unittest

from scripts.run_agent_evaluation import latency_percentiles, synthetic_markdown


class PerformanceEvaluationContractTests(unittest.TestCase):
    def test_latency_percentiles_are_deterministic(self):
        values = list(range(1, 101))

        self.assertEqual(
            latency_percentiles(values),
            {"p50": 50.0, "p95": 95.0, "p99": 99.0, "max": 100.0},
        )

    def test_latency_percentiles_handle_empty_input(self):
        self.assertEqual(
            latency_percentiles([]),
            {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
        )

    def test_synthetic_corpus_has_stable_unique_markers(self):
        text = synthetic_markdown(3, 4).decode("utf-8")

        self.assertEqual(text.count("## Record"), 4)
        self.assertIn("PERF-3-0", text)
        self.assertIn("PERF-3-3", text)
        self.assertIn("status blocked", text)
        self.assertIn("status active", text)


if __name__ == "__main__":
    unittest.main()
