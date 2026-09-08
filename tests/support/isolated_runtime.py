"""Isolate backend tests from the developer's databases, uploads, and Chroma data."""

from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterator
from unittest.mock import patch


@contextmanager
def isolated_runtime(*, hide_zai_key: bool = True) -> Iterator[Path]:
    """Patch every imported persistence constant to a disposable test root."""

    root = Path(tempfile.mkdtemp(prefix="chatbot-agent-rag-test-"))
    data_dir = root / "data"
    db_path = root / "chat_meta.test.db"
    history_url = f"sqlite:///{(root / 'history.test.sqlite').as_posix()}"

    import config
    import backend.main as api_main
    import core.document_processor as document_processor
    import core.history as history
    import core.persistent_storage as persistent_storage
    import core.retriever as retriever
    import core.skill_workflow as skill_workflow
    import core.trace as trace
    import db.file_manager as file_manager
    import db.sandbox_manager as sandbox_manager
    import db.session_manager as session_manager
    import db.skill_manager as skill_manager

    targets = [
        (config, "DATA_DIR", str(data_dir)),
        (config, "DB_NAME", str(db_path)),
        (config, "HISTORY_DB_PATH", history_url),
        (api_main, "DATA_DIR", str(data_dir)),
        (document_processor, "DATA_DIR", str(data_dir)),
        (persistent_storage, "DATA_DIR", str(data_dir)),
        (retriever, "DATA_DIR", str(data_dir)),
        (skill_workflow, "DATA_DIR", str(data_dir)),
        (history, "HISTORY_DB_PATH", history_url),
        (trace, "DB_NAME", str(db_path)),
        (file_manager, "DB_NAME", str(db_path)),
        (sandbox_manager, "DB_NAME", str(db_path)),
        (session_manager, "DB_NAME", str(db_path)),
        (skill_manager, "DB_NAME", str(db_path)),
    ]

    old_store = persistent_storage._store
    old_key = os.environ.get("ZAI_API_KEY")
    with ExitStack() as stack:
        for module, name, value in targets:
            stack.enter_context(patch.object(module, name, value))
        persistent_storage._store = persistent_storage.PersistentFileStore(str(data_dir))
        if hide_zai_key:
            os.environ.pop("ZAI_API_KEY", None)
        try:
            session_manager.init_meta_db()
            file_manager.init_file_db()
            trace.init_trace_db()
            skill_manager.init_skill_db()
            sandbox_manager.init_sandbox_db()
            retriever.invalidate_retriever_cache()
            yield root
        finally:
            retriever.invalidate_retriever_cache()
            persistent_storage._store = old_store
            if old_key is None:
                os.environ.pop("ZAI_API_KEY", None)
            else:
                os.environ["ZAI_API_KEY"] = old_key
    shutil.rmtree(root, ignore_errors=True)


def fixture_root() -> Path:
    return Path(__file__).resolve().parents[1] / "fixtures" / "agent_rag"
