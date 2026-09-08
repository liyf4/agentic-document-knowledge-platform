import os
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from tests.support.isolated_runtime import isolated_runtime


class FakeAgent:
    def __init__(self, output="Backend agent response"):
        self.output = output

    def invoke(self, *_args, **_kwargs):
        return {"output": self.output, "intermediate_steps": []}


class BackendApiTests(unittest.TestCase):
    def test_session_chat_and_trace_are_available_without_frontend(self):
        with isolated_runtime(), patch.dict(os.environ, {"ZAI_API_KEY":"offline-test-key"}, clear=False), patch(
            "backend.main.create_agent_executor", return_value=FakeAgent("Hello from the backend agent")
        ):
            with TestClient(app) as client:
                created = client.post("/api/sessions")
                self.assertEqual(200, created.status_code)
                session_id = created.json()["id"]
                response = client.post("/api/chat", json={"session_id":session_id,"message":"Hello","mode":"chat"})
                self.assertEqual(200, response.status_code, response.text)
                payload = response.json()
                self.assertEqual("general_chat", payload["retrieval_debug"]["route"])
                self.assertEqual([], payload["retrieval_debug"]["route_violation"])
                traces = client.get(f"/api/sessions/{session_id}/traces").json()["traces"]
                self.assertEqual(1, len(traces))
                self.assertEqual("Hello from the backend agent", traces[0]["answer"])

    def test_upload_processing_runs_off_event_loop_and_returns_sync_mode(self):
        def slow_process(_upload, _session):
            time.sleep(0.05)
            return True, "indexed"

        with isolated_runtime(), patch("backend.main.process_file", side_effect=slow_process):
            with TestClient(app) as client:
                session_id = client.post("/api/sessions").json()["id"]
                response = client.post(
                    f"/api/sessions/{session_id}/documents",
                    files={"file":("knowledge.md", b"# Knowledge\nEvidence", "text/markdown")},
                )
        self.assertEqual(200, response.status_code, response.text)
        self.assertEqual("sync", response.json()["mode"])

    def test_missing_session_is_rejected(self):
        with isolated_runtime():
            with TestClient(app) as client:
                response = client.get("/api/sessions/not-a-session/documents")
        self.assertEqual(404, response.status_code)


if __name__ == "__main__":
    unittest.main()
