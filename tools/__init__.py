__all__ = ["web_search", "read_website", "get_current_time"]


def __getattr__(name):
    if name in {"web_search", "read_website"}:
        from .web_search import read_website, web_search

        return {"web_search": web_search, "read_website": read_website}[name]
    if name == "get_current_time":
        from .system import get_current_time

        return get_current_time
    raise AttributeError(f"module 'tools' has no attribute {name!r}")
