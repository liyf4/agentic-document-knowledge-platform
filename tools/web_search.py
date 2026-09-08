"""Structured web search and page verification tools."""

from __future__ import annotations

import hashlib
import json
import os
import re
import ssl
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib.parse import urlparse

os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36")

import certifi
import requests
import urllib3
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_WEB_TIMEOUT = (5, 20)
_WEB_HEADERS = {
    "User-Agent": os.environ["USER_AGENT"],
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class _SSLContextAdapter(HTTPAdapter):
    def __init__(self, ssl_context: ssl.SSLContext, **kwargs: Any):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False, **pool_kwargs: Any):
        pool_kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def proxy_manager_for(self, proxy: str, **proxy_kwargs: Any):
        proxy_kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def _safe_error(error: Exception) -> str:
    return re.sub(r"([?&](?:key|api_key)=)[^&\s>]+", r"\1<redacted>", str(error), flags=re.IGNORECASE)


def _proxy_url() -> str:
    proxy_port = os.getenv("PROXY_PORT")
    return f"http://127.0.0.1:{proxy_port}" if proxy_port else ""


def _create_session(*, use_proxy: bool) -> requests.Session:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    session = requests.Session()
    session.headers.update(_WEB_HEADERS)
    session.trust_env = False
    session.mount("https://", _SSLContextAdapter(ssl_context))
    if use_proxy and _proxy_url():
        session.proxies.update({"http": _proxy_url(), "https": _proxy_url()})
    return session


def _candidate(title: str, url: str, snippet: str, provider: str) -> Dict[str, str]:
    return {"title": title or "Untitled", "url": url, "snippet": snippet or "", "provider": provider}


def search_web_candidates(query: str) -> Dict[str, Any]:
    """Search only for candidate URLs. Snippets are explicitly unverified."""
    google_key, google_cx = os.getenv("GOOGLE_API_KEY"), os.getenv("GOOGLE_CSE_ID")
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    errors: List[Dict[str, str]] = []

    if google_key and google_cx:
        try:
            response = _create_session(use_proxy=bool(_proxy_url())).get(
                "https://customsearch.googleapis.com/customsearch/v1",
                params={"q": query, "cx": google_cx, "num": 5, "key": google_key}, timeout=_WEB_TIMEOUT,
            )
            response.raise_for_status()
            items = response.json().get("items") or []
            candidates = [_candidate(i.get("title", ""), i.get("link", ""), i.get("snippet", ""), "google") for i in items]
            if candidates:
                return {"ok": True, "query": query, "candidates": candidates, "evidence_status": "unverified_search_candidates"}
            errors.append({"error_code": "NO_SEARCH_RESULTS", "message": "Google returned no results."})
        except Exception as exc:
            errors.append({"error_code": "GOOGLE_SEARCH_FAILED", "message": _safe_error(exc)})

    if serpapi_key:
        try:
            response = _create_session(use_proxy=bool(_proxy_url())).get(
                "https://serpapi.com/search.json",
                params={"engine": "google", "google_domain": "google.com", "hl": "zh-cn", "q": query, "api_key": serpapi_key},
                timeout=_WEB_TIMEOUT,
            )
            response.raise_for_status()
            data = response.json()
            if data.get("error"):
                raise RuntimeError(str(data["error"]))
            items = data.get("organic_results") or []
            candidates = [_candidate(i.get("title", ""), i.get("link", ""), i.get("snippet", ""), "serpapi") for i in items[:5]]
            if candidates:
                return {"ok": True, "query": query, "candidates": candidates, "evidence_status": "unverified_search_candidates"}
            errors.append({"error_code": "NO_SEARCH_RESULTS", "message": "SerpAPI returned no results."})
        except Exception as exc:
            errors.append({"error_code": "SERPAPI_SEARCH_FAILED", "message": _safe_error(exc)})

    error_code = "WEB_SEARCH_NOT_CONFIGURED" if not (google_key and google_cx) and not serpapi_key else "WEB_SEARCH_FAILED"
    return {"ok": False, "error_code": error_code, "errors": errors, "candidates": []}


def _meta_content(soup: BeautifulSoup, *names: str) -> str:
    for name in names:
        element = soup.find("meta", attrs={"name": name}) or soup.find("meta", attrs={"property": name})
        if element and element.get("content"):
            return str(element.get("content")).strip()
    return ""


def _source_type(url: str) -> str:
    host = urlparse(url).netloc.casefold().split(":")[0]
    if host.endswith(".gov") or ".gov." in host:
        return "government"
    if host.endswith(".edu") or ".edu." in host:
        return "academic"
    if host == "owasp.org" or host.endswith(".owasp.org") or host in {
        "www.w3.org", "w3.org", "nvlpubs.nist.gov", "nist.gov", "www.nist.gov",
    }:
        return "standards_or_nonprofit_authority"
    if any(part in host for part in ("microsoft.com", "google.com", "mozilla.org", "cloudflare.com")):
        return "vendor_primary_documentation"
    return "web_page"


def read_web_page(url: str) -> Dict[str, Any]:
    """Open and extract a page. Only a successful result can become verified evidence."""
    if urlparse(url).scheme not in {"http", "https"}:
        return {"ok": False, "error_code": "INVALID_URL_SCHEME", "url": url}
    first_error = ""
    response = None
    for use_proxy in ([True, False] if _proxy_url() else [False]):
        try:
            response = _create_session(use_proxy=use_proxy).get(url, timeout=_WEB_TIMEOUT)
            response.raise_for_status()
            break
        except Exception as exc:
            if not first_error:
                first_error = _safe_error(exc)
            response = None
    if response is None:
        return {"ok": False, "error_code": "WEB_PAGE_OPEN_FAILED", "url": url, "error": first_error}

    soup = BeautifulSoup(response.text, "html.parser")
    title = (_meta_content(soup, "og:title", "twitter:title") or (soup.title.get_text(" ", strip=True) if soup.title else "Untitled"))
    published = _meta_content(soup, "article:published_time", "datePublished", "date", "pubdate")
    updated = _meta_content(soup, "article:modified_time", "dateModified", "last-modified")
    for element in soup(["script", "style", "noscript", "nav", "footer"]):
        element.decompose()
    text = " ".join(soup.get_text(" ", strip=True).split())[:8000]
    final_url = str(response.url)
    accessed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    evidence_id = "web_" + hashlib.sha256((final_url + "|" + text[:1000]).encode("utf-8")).hexdigest()[:16]
    return {
        "ok": bool(text), "error_code": "" if text else "EMPTY_WEB_PAGE", "kind": "verified_web",
        "evidence_id": evidence_id, "title": title, "url": final_url,
        "published_at": published, "updated_at": updated, "accessed_at": accessed_at,
        "text": text, "source_type": _source_type(final_url),
    }


@tool
def web_search(query: str) -> str:
    """Search the web for candidate sources; returned snippets are not verified evidence."""
    return json.dumps(search_web_candidates(query), ensure_ascii=False, default=str)


@tool
def read_website(url: str) -> str:
    """Open and extract a web page as structured, verified evidence."""
    return json.dumps(read_web_page(url), ensure_ascii=False, default=str)
