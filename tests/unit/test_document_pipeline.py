import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from langchain_core.documents import Document

from core.document_analysis import build_document_analysis, extract_rule_facts
from core.document_processor import DocumentProcessor
from core.document_loaders.dispatcher import load_documents
from core.query_transform import generate_hyde_query


class DocumentLoaderTests(unittest.TestCase):
    def test_markdown_preserves_section_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "knowledge.md"
            path.write_text("# Root\n## Atlas\nPayment API.\n## Beacon\nWarehouse tables.", encoding="utf-8")
            docs = load_documents(str(path), path.name)
        paths = [str(item.metadata.get("section_path")) for item in docs]
        self.assertTrue(any("Atlas" in item for item in paths))
        self.assertTrue(any("Beacon" in item for item in paths))
        self.assertTrue(all(item.metadata.get("pre_chunked") for item in docs))

    def test_csv_produces_summary_and_row_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "actions.csv"
            pd.DataFrame([{"task_id":"AT-101","owner":"Alice"},{"task_id":"AT-102","owner":"Bob"}]).to_csv(path, index=False)
            docs = load_documents(str(path), path.name)
        kinds = {item.metadata.get("chunk_kind") for item in docs}
        self.assertIn("table_summary", kinds)
        self.assertIn("table_rows", kinds)
        self.assertTrue(any("AT-101" in item.page_content for item in docs))

    def test_xlsx_produces_sheet_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "budget.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame([{"project":"Atlas","budget":120000}]).to_excel(writer, sheet_name="Budget", index=False)
            docs = load_documents(str(path), path.name)
        self.assertTrue(docs)
        self.assertTrue(all(item.metadata.get("sheet_name") == "Budget" for item in docs))


class MetadataTests(unittest.TestCase):
    def test_metadata_timeout_is_classified_without_hiding_index_success(self):
        self.assertEqual("llm_timeout", DocumentProcessor._metadata_failure_status(TimeoutError("metadata timed out")))

    def test_rule_facts_keep_source_and_evidence(self):
        docs = [Document(
            page_content="Approved budget USD 120,000. Review date 2025-06-25. Contact owner@example.com.",
            metadata={"file_id":"f1","source":"knowledge.md","chunk_id":"c1","section_path":"Facts"},
        )]
        facts = extract_rule_facts(docs)
        fact_types = {item["fact_type"] for item in facts}
        self.assertTrue({"amount", "date", "email"}.issubset(fact_types))
        self.assertTrue(all(item["chunk_id"] == "c1" for item in facts))
        self.assertTrue(all(item["section_path"] == "Facts" for item in facts))
        self.assertTrue(all(item["evidence_text"] for item in facts))

    def test_document_analysis_keeps_blocks_and_structure(self):
        docs = [Document(page_content="Atlas evidence", metadata={"source":"kb.md","chunk_id":"c1","chunk_kind":"text","section_path":"Atlas"})]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "kb.md"
            path.write_text("Atlas evidence", encoding="utf-8")
            metadata, _facts, blocks = build_document_analysis(
                docs, file_path=str(path), file_name="kb.md", file_id="f1", session_id="s1", include_llm=False
            )
        self.assertEqual(1, len(blocks))
        self.assertEqual("Atlas", blocks[0]["section_path"])
        self.assertEqual("kb", metadata["title"])


class QueryTransformTests(unittest.TestCase):
    @patch("core.query_transform.ENABLE_HYDE", True)
    @patch.dict("os.environ", {}, clear=True)
    def test_missing_api_key_falls_back(self):
        query, debug = generate_hyde_query("Atlas recovery")
        self.assertEqual("Atlas recovery", query)
        self.assertEqual("missing_api_key", debug["status"])

    @patch("core.query_transform.ENABLE_HYDE", True)
    @patch("core.query_transform.HYDE_MAX_QUERY_LENGTH", 5)
    def test_long_query_is_skipped(self):
        query, debug = generate_hyde_query("a query longer than five characters")
        self.assertEqual("skipped_long_query", debug["status"])
        self.assertEqual("a query longer than five characters", query)

    @patch("core.query_transform.ENABLE_HYDE", True)
    @patch.dict("os.environ", {"ZAI_API_KEY":"test"}, clear=False)
    def test_generated_hyde_is_appended_and_bounded(self):
        class Response:
            content = "hypothetical evidence " * 30
        class LLM:
            def invoke(self, _prompt):
                return Response()
        with patch("core.query_transform.ComponentFactory.get_component", return_value=LLM()):
            query, debug = generate_hyde_query("Atlas recovery")
        self.assertEqual("generated", debug["status"])
        self.assertTrue(query.startswith("Atlas recovery\n"))
        self.assertLessEqual(len(debug["hyde_query"]), 180)


if __name__ == "__main__":
    unittest.main()
