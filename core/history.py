from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from typing import List, Sequence

from config import HISTORY_DB_PATH


def _async_sqlite_url(connection: str) -> str:
    if connection.startswith("sqlite+aiosqlite:"):
        return connection
    if connection.startswith("sqlite:"):
        return connection.replace("sqlite:", "sqlite+aiosqlite:", 1)
    return connection


def get_session_history(session_id: str):
    """
    使用 SQLChatMessageHistory 持久化存储聊天记录。
    它会自动在 sqlite 数据库中创建 message_store 表。
    """
    try:
        return SQLChatMessageHistory(session_id=session_id, connection=HISTORY_DB_PATH)
    except TypeError:
        return SQLChatMessageHistory(session_id=session_id, connection_string=HISTORY_DB_PATH)


def get_async_session_history(session_id: str):
    """
    Async variant for streaming endpoints that call LangChain async APIs.
    """
    connection = _async_sqlite_url(HISTORY_DB_PATH)
    try:
        return SQLChatMessageHistory(session_id=session_id, connection=connection, async_mode=True)
    except TypeError:
        return SQLChatMessageHistory(session_id=session_id, connection_string=connection, async_mode=True)


class WindowedChatMessageHistory(BaseChatMessageHistory):
    """Expose a bounded history view while delegating writes to durable full history."""

    def __init__(self, delegate: BaseChatMessageHistory, visible_messages: Sequence[BaseMessage]):
        self._delegate = delegate
        self._visible_messages = list(visible_messages)

    @property
    def messages(self) -> List[BaseMessage]:
        return list(self._visible_messages)

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        self._delegate.add_messages(list(messages))

    def clear(self) -> None:
        self._delegate.clear()

    async def aget_messages(self) -> List[BaseMessage]:
        return list(self._visible_messages)

    async def aadd_messages(self, messages: Sequence[BaseMessage]) -> None:
        if hasattr(self._delegate, "aadd_messages"):
            await self._delegate.aadd_messages(list(messages))
        else:
            self._delegate.add_messages(list(messages))

    async def aclear(self) -> None:
        if hasattr(self._delegate, "aclear"):
            await self._delegate.aclear()
        else:
            self._delegate.clear()


def get_windowed_session_history(
    session_id: str, visible_messages: Sequence[BaseMessage], *, async_mode: bool = False
) -> WindowedChatMessageHistory:
    delegate = get_async_session_history(session_id) if async_mode else get_session_history(session_id)
    return WindowedChatMessageHistory(delegate, visible_messages)
