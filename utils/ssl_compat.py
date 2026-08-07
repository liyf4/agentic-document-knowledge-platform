from __future__ import annotations

import ssl

try:
    import certifi
except Exception:  # pragma: no cover - best-effort environment compatibility.
    certifi = None


_ORIGINAL_CREATE_DEFAULT_CONTEXT = ssl.create_default_context
_PATCHED = False


def prefer_certifi_default_context() -> None:
    """Avoid broken Windows certificate-store entries in local conda envs."""

    global _PATCHED
    if _PATCHED or certifi is None:
        return

    def create_default_context(
        purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH,
        *,
        cafile: str | None = None,
        capath: str | None = None,
        cadata: str | bytes | None = None,
    ) -> ssl.SSLContext:
        if cafile is None and capath is None and cadata is None:
            cafile = certifi.where()
        return _ORIGINAL_CREATE_DEFAULT_CONTEXT(
            purpose=purpose,
            cafile=cafile,
            capath=capath,
            cadata=cadata,
        )

    ssl.create_default_context = create_default_context
    _PATCHED = True
