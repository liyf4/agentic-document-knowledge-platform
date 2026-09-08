"""Long-conversation token budget and persistence tests."""

import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

import core.context_manager as context
from core.history import WindowedChatMessageHistory


class _History:
    def __init__(self, messages):
        self.messages = list(messages)
        self.added = []

    def add_messages(self, messages):
        self.added.extend(messages)

    def clear(self):
        self.messages.clear()


class TokenBudgetTests(unittest.TestCase):
    def _state(self):
        return {
            "conversation_summary": "",
            "summarized_message_count": 0,
            "context_usage": {},
            "last_compacted_at": None,
        }

    def test_many_short_turns_do_not_compact_when_token_budget_is_low(self):
        messages = []
        for index in range(40):
            messages.extend([HumanMessage(content=f"问{index}"), AIMessage(content="答")])
        with patch.object(context, "LLM_CONTEXT_WINDOW_TOKENS", 10000), patch.object(
            context, "LLM_OUTPUT_RESERVE_TOKENS", 100
        ), patch.object(context, "CONTEXT_TOOL_GROWTH_RESERVE_TOKENS", 100), patch.object(
            context, "get_conversation_context_state", return_value=self._state()
        ), patch.object(context, "get_session_history", return_value=_History(messages)), patch.object(
            context, "_generate_summary"
        ) as summarize:
            plan = context.prepare_session_context("short-session", "继续")
        self.assertFalse(plan.compacted)
        self.assertEqual(messages, plan.visible_messages)
        summarize.assert_not_called()

    def test_large_history_compacts_to_token_target_without_deleting_messages(self):
        messages = []
        for index in range(8):
            messages.extend([
                HumanMessage(content=f"任务 {index} " + "甲" * 700),
                AIMessage(content="结论 " + "乙" * 700),
            ])
        stored = _History(messages)
        updates = []
        with patch.object(context, "LLM_CONTEXT_WINDOW_TOKENS", 5000), patch.object(
            context, "LLM_OUTPUT_RESERVE_TOKENS", 100
        ), patch.object(context, "CONTEXT_TOOL_GROWTH_RESERVE_TOKENS", 100), patch.object(
            context, "CONTEXT_SUMMARY_MAX_TOKENS", 200
        ), patch.object(context, "get_conversation_context_state", return_value=self._state()), patch.object(
            context, "get_session_history", return_value=stored
        ), patch.object(context, "_generate_summary", return_value="压缩摘要"), patch.object(
            context, "update_conversation_summary", side_effect=lambda *args: updates.append(args)
        ):
            plan = context.prepare_session_context("large-session", "继续执行")
        self.assertTrue(plan.compacted)
        self.assertGreater(plan.compacted_message_count, 0)
        self.assertLessEqual(plan.estimated_prompt_tokens, int(5000 * 0.30))
        self.assertEqual(messages, stored.messages)
        self.assertEqual(plan.compacted_message_count, updates[0][2])

    def test_compaction_failure_below_hard_limit_degrades_with_warning(self):
        messages = [HumanMessage(content="甲" * 2100), AIMessage(content="乙" * 2100)]
        with patch.object(context, "LLM_CONTEXT_WINDOW_TOKENS", 3500), patch.object(
            context, "LLM_OUTPUT_RESERVE_TOKENS", 100
        ), patch.object(context, "CONTEXT_TOOL_GROWTH_RESERVE_TOKENS", 100), patch.object(
            context, "CONTEXT_SUMMARY_MAX_TOKENS", 100
        ), patch.object(context, "get_conversation_context_state", return_value=self._state()), patch.object(
            context, "get_session_history", return_value=_History(messages)
        ), patch.object(context, "_generate_summary", side_effect=RuntimeError("summary unavailable")):
            plan = context.prepare_session_context("warning-session", "继续")
        self.assertFalse(plan.compacted)
        self.assertIn("summary unavailable", plan.warning)

    def test_compaction_failure_at_hard_limit_blocks_request(self):
        messages = [HumanMessage(content="甲" * 2300), AIMessage(content="乙" * 2300)]
        with patch.object(context, "LLM_CONTEXT_WINDOW_TOKENS", 3500), patch.object(
            context, "LLM_OUTPUT_RESERVE_TOKENS", 100
        ), patch.object(context, "CONTEXT_TOOL_GROWTH_RESERVE_TOKENS", 100), patch.object(
            context, "CONTEXT_SUMMARY_MAX_TOKENS", 100
        ), patch.object(context, "get_conversation_context_state", return_value=self._state()), patch.object(
            context, "get_session_history", return_value=_History(messages)
        ), patch.object(context, "_generate_summary", side_effect=RuntimeError("summary unavailable")):
            with self.assertRaises(context.ContextLimitError):
                context.prepare_session_context("blocked-session", "继续")


class WindowedHistoryTests(unittest.TestCase):
    def test_windowed_history_reads_tail_and_writes_to_full_history(self):
        delegate = _History([HumanMessage(content="old")])
        tail = [HumanMessage(content="recent")]
        view = WindowedChatMessageHistory(delegate, tail)
        self.assertEqual(tail, view.messages)
        added = [AIMessage(content="new")]
        view.add_messages(added)
        self.assertEqual(added, delegate.added)


class SessionPersistenceTests(unittest.TestCase):
    def test_context_state_migrates_and_session_delete_clears_chat_history(self):
        import db.session_manager as sessions

        durable_history = _History([HumanMessage(content="old")])
        with tempfile.TemporaryDirectory() as directory, patch.object(
            sessions, "DB_NAME", str(Path(directory) / "meta.sqlite")
        ), patch("core.history.get_session_history", return_value=durable_history):
            sessions.init_meta_db()
            session_id = sessions.create_new_session("context test")
            sessions.update_conversation_summary(session_id, "摘要", 4, "now")
            sessions.update_context_usage(session_id, {"usage_percent": 30.0})
            state = sessions.get_conversation_context_state(session_id)
            self.assertEqual("摘要", state["conversation_summary"])
            self.assertEqual(4, state["summarized_message_count"])
            self.assertEqual(30.0, state["context_usage"]["usage_percent"])
            sessions.delete_session_data(session_id)
            self.assertEqual([], durable_history.messages)
            self.assertEqual("", sessions.get_conversation_context_state(session_id)["conversation_summary"])


if __name__ == "__main__":
    unittest.main()
