import unittest
from unittest.mock import patch

from langchain_core.documents import Document

from core.agent import _bounded_tool_observation, build_document_evidence, group_document_evidence
from core.orchestration import WorkerResult, _supervisor_context, _verify
from core.request_router import RouteDecision, find_route_violations, route_chat_request


def decision(**overrides):
    values = dict(
        route="orchestrated", allowed_tools=["retrieve_knowledge", "query_document_metadata"],
        blocked_tools=[], reason="test", matched_skill_ids=[], needs_documents=True,
        needs_web=False, needs_workspace=False, needs_structured_facts=False, use_multi_agent=True,
    )
    values.update(overrides)
    return RouteDecision(**values)


class EvidenceTests(unittest.TestCase):
    def test_evidence_ids_are_stable_and_sections_do_not_merge(self):
        hits = [
            (0.9, Document(page_content="Atlas payment API", metadata={"file_id":"f1","source":"kb.md","section_path":"Atlas","chunk_id":"c1","chunk_kind":"text"})),
            (0.8, Document(page_content="Beacon warehouse", metadata={"file_id":"f1","source":"kb.md","section_path":"Beacon","chunk_id":"c2","chunk_kind":"text"})),
        ]
        first = build_document_evidence(hits)
        second = build_document_evidence(hits)
        self.assertEqual([x["evidence_id"] for x in first], [x["evidence_id"] for x in second])
        self.assertEqual(2, len(group_document_evidence(first)))

    def test_tool_observation_is_bounded(self):
        rendered = _bounded_tool_observation("x" * 100_000)
        self.assertLess(len(rendered), 60_000)
        self.assertIn("truncated for context budget", rendered)


class SupervisorTests(unittest.TestCase):
    def test_completed_capability_is_removed(self):
        context = _supervisor_context(decision(), [WorkerResult("document", "completed", completed_capabilities=["retrieve_knowledge"])])
        self.assertNotIn("retrieve_knowledge", context.remaining_tools)

    def test_failed_capability_gets_one_bounded_retry(self):
        context = _supervisor_context(decision(), [WorkerResult("document", "failed")])
        self.assertEqual(1, context.tool_call_limits["retrieve_knowledge"])

    def test_verification_reports_missing_evidence(self):
        result = _verify([WorkerResult("document", "failed")])
        self.assertTrue(any("No document" in item for item in result.uncertainties))


class RoutingTests(unittest.TestCase):
    @patch("core.request_router.match_skills_with_workflow_fallback", return_value=[])
    @patch("core.request_router.get_session_documents", return_value=[])
    def test_general_chat_route(self, *_mocks):
        result = route_chat_request("s1", "Hello")
        self.assertEqual("general_chat", result.route)
        self.assertEqual(["get_current_time"], result.allowed_tools)

    @patch("core.request_router.match_skills_with_workflow_fallback", return_value=[])
    @patch("core.request_router.get_session_documents", return_value=[("f1", "kb.md", 3, "completed", "", "")])
    def test_document_route_allows_only_retrieval(self, *_mocks):
        result = route_chat_request("s1", "Summarize the uploaded document with citations")
        self.assertEqual("document_qa", result.route)
        self.assertEqual(["retrieve_knowledge"], result.allowed_tools)

    def test_route_violation_is_detected(self):
        self.assertEqual(["run_python_code"], find_route_violations("document_qa", [{"tool":"run_python_code"}]))


if __name__ == "__main__":
    unittest.main()
