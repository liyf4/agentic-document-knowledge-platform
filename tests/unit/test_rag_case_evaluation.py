import unittest

from scripts.run_rag_case_evaluation import (
    FIXTURE_DIR,
    extract_evidence_ids,
    load_cases,
    score_answer,
)


class RagCaseEvaluationTests(unittest.TestCase):
    def test_fixed_case_set_is_complete_and_unique(self):
        cases = load_cases(FIXTURE_DIR / "cases.jsonl")

        self.assertEqual(len(cases), 50)
        self.assertEqual(len({case["id"] for case in cases}), 50)

    def test_every_gold_evidence_id_exists_in_documents(self):
        corpus = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (FIXTURE_DIR / "documents").iterdir()
            if path.is_file()
        )
        available = set(extract_evidence_ids(corpus))
        required = {
            evidence_id
            for case in load_cases(FIXTURE_DIR / "cases.jsonl")
            for evidence_id in case["evidence_ids"]
        }

        self.assertTrue(required.issubset(available), required - available)

    def test_answer_accuracy_uses_fact_groups(self):
        result = score_answer(
            "Priya Nair approved a USD 185,000 ceiling.",
            [["Priya Nair"], ["185,000", "185000"], ["2026-03-08"]],
        )

        self.assertEqual(result["accuracy"], 0.6667)
        self.assertEqual(result["missing_checks"], [["2026-03-08"]])


if __name__ == "__main__":
    unittest.main()
