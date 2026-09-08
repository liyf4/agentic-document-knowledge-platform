import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from core.retriever import (
    HybridRRFRetriever,
    _rrf_fuse_with_scores,
    _should_skip_rerank,
    plan_retrieval,
    retrieve_with_rerank_debug,
)


def doc(text, chunk_id, tier="primary"):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "source": "fixture.md", "retrieval_tier": tier})


class FakeRetriever:
    def __init__(self, documents):
        self.documents = documents

    def invoke(self, _query):
        return list(self.documents)


class RetrieverComponentTests(unittest.TestCase):
    def test_rrf_deduplicates_by_chunk_identity(self):
        a, b = doc("alpha", "a"), doc("beta", "b")
        ranked, scores = _rrf_fuse_with_scores([a, b], [a])
        self.assertEqual(["a", "b"], [item.metadata["chunk_id"] for _, item in ranked])
        a_key = next(key for key in scores if "|a|" in key)
        b_key = next(key for key in scores if "|b|" in key)
        self.assertGreater(scores[a_key], scores[b_key])

    def test_default_plan_excludes_secondary_material(self):
        plan = plan_retrieval("Explain the project decision")
        self.assertEqual("primary", plan.scope)

    def test_exact_identifier_can_skip_rerank(self):
        candidates = [(0.04 - index / 1000, doc(str(index), str(index))) for index in range(6)]
        skipped, reason = _should_skip_rerank("AT-103", candidates)
        self.assertTrue(skipped)
        self.assertEqual("short_exact_lookup_query", reason)

    def test_complex_query_executes_reranker(self):
        documents = [doc(f"evidence {index}", str(index)) for index in range(6)]
        hybrid = HybridRRFRetriever(FakeRetriever(documents), FakeRetriever(list(reversed(documents))))
        fake_scores = [(1.0 - index / 10, item) for index, item in enumerate(documents)]
        with patch("core.retriever.rerank", return_value=fake_scores) as rerank:
            final, debug = retrieve_with_rerank_debug(
                "Compare the recovery decisions for Atlas and Beacon and explain their consequences", hybrid
            )
        rerank.assert_called_once()
        self.assertFalse(debug["rerank_skipped"])
        self.assertGreater(debug["rerank_count"], 0)
        self.assertTrue(final)

    def test_hybrid_debug_exposes_all_retrieval_stages(self):
        documents = [doc(f"evidence {index}", str(index)) for index in range(6)]
        hybrid = HybridRRFRetriever(FakeRetriever(documents), FakeRetriever(documents))
        with patch("core.retriever.rerank", return_value=[(1.0, item) for item in documents]):
            _, debug = retrieve_with_rerank_debug("Explain this complex architecture decision", hybrid)
        self.assertTrue(debug["dense"])
        self.assertTrue(debug["bm25"])
        self.assertTrue(debug["rrf"])
        for stage in ("dense", "bm25", "rrf", "rerank", "total"):
            self.assertIn(stage, debug["timing_ms"])


if __name__ == "__main__":
    unittest.main()
