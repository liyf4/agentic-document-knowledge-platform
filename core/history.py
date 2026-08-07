from langchain_community.chat_message_histories import SQLChatMessageHistory

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
