import os
import re
import ssl

os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0 Safari/537.36",
)

import certifi
import requests
import urllib3
from bs4 import BeautifulSoup
from langchain_core.tools import tool
from requests.adapters import HTTPAdapter

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_WEB_TIMEOUT = (5, 20)
_WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


class _SSLContextAdapter(HTTPAdapter):
    def __init__(self, ssl_context: ssl.SSLContext, **kwargs):
        self._ssl_context = ssl_context
        super().__init__(**kwargs)

    def init_poolmanager(self, connections, maxsize, block=False, **pool_kwargs):
        pool_kwargs["ssl_context"] = self._ssl_context
        return super().init_poolmanager(connections, maxsize, block=block, **pool_kwargs)

    def proxy_manager_for(self, proxy, **proxy_kwargs):
        proxy_kwargs["ssl_context"] = self._ssl_context
        return super().proxy_manager_for(proxy, **proxy_kwargs)


def _safe_error(error: Exception) -> str:
    return re.sub(r"([?&]key=)[^&\s>]+", r"\1<redacted>", str(error))


def _is_google_api_config_error(message: str) -> bool:
    return (
        ("customsearch.googleapis.com" in message or "customsearch.googleapis.com/customsearch" in message)
        and (
            "403 Client Error" in message
            or "accessNotConfigured" in message
            or "blocked" in message
            or "forbidden" in message
            or "has not been used" in message
            or "it is disabled" in message
        )
    )


def _google_config_message() -> str:
    return (
        "Google 搜索配置不可用: Google Custom Search API 返回 403。"
        "请在当前 GOOGLE_API_KEY 所属的 Google Cloud 项目中启用 "
        "Custom Search API，并确认 API key 允许调用 customsearch.googleapis.com。"
        "也可以改用 SERPAPI_API_KEY 作为备用搜索源。"
    )


def _proxy_url() -> str:
    proxy_port = os.getenv("PROXY_PORT")
    return f"http://127.0.0.1:{proxy_port}" if proxy_port else ""


def _create_session(*, use_proxy: bool) -> requests.Session:
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    session = requests.Session()
    session.headers.update(_WEB_HEADERS)
    session.trust_env = False
    session.mount("https://", _SSLContextAdapter(ssl_context))

    proxy_url = _proxy_url()
    if use_proxy and proxy_url:
        session.proxies.update({"http": proxy_url, "https": proxy_url})
    return session


def _load_website(url: str, *, use_proxy: bool) -> str:
    session = _create_session(use_proxy=use_proxy)
    response = session.get(url, timeout=_WEB_TIMEOUT)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for element in soup(["script", "style", "noscript"]):
        element.decompose()
    content = soup.get_text(" ", strip=True)
    cleaned_content = " ".join(content.split())[:8000]
    return f"网页内容预览:\n{cleaned_content}"


def _serpapi_search(query: str) -> str:
    session = _create_session(use_proxy=bool(_proxy_url()))
    response = session.get(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "google_domain": "google.com",
            "hl": "zh-cn",
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
        },
        timeout=_WEB_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        return f"SerpAPI 搜索失败: {data['error']}"

    results = data.get("organic_results") or []
    if not results:
        answer_box = data.get("answer_box") or {}
        answer = answer_box.get("answer") or answer_box.get("snippet")
        return answer or "SerpAPI 未返回搜索结果。"

    lines = []
    for index, item in enumerate(results[:5], start=1):
        title = item.get("title") or "Untitled"
        link = item.get("link") or ""
        snippet = item.get("snippet") or ""
        lines.append(f"{index}. {title}\n{link}\n{snippet}".strip())
    return "\n\n".join(lines)


def _google_search(query: str) -> str:
    session = _create_session(use_proxy=bool(_proxy_url()))
    response = session.get(
        "https://customsearch.googleapis.com/customsearch/v1",
        params={
            "q": query,
            "cx": os.getenv("GOOGLE_CSE_ID"),
            "num": 5,
            "key": os.getenv("GOOGLE_API_KEY"),
        },
        timeout=_WEB_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    results = data.get("items") or []
    if not results:
        return "Google 搜索未返回结果。"

    lines = []
    for index, item in enumerate(results[:5], start=1):
        title = item.get("title") or "Untitled"
        link = item.get("link") or ""
        snippet = item.get("snippet") or ""
        lines.append(f"{index}. {title}\n{link}\n{snippet}".strip())
    return "\n\n".join(lines)


@tool
def web_search(query: str) -> str:
    """Search the web for real-time internet or external public information."""
    google_error = ""
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")
    serpapi_api_key = os.getenv("SERPAPI_API_KEY")

    if google_api_key and google_cse_id:
        try:
            return _google_search(query)
        except Exception as exc:
            google_error = _safe_error(exc)
            if not _is_google_api_config_error(google_error):
                return f"Google 搜索失败: {google_error}"

    if serpapi_api_key:
        try:
            return _serpapi_search(query)
        except Exception as exc:
            serpapi_error = _safe_error(exc)
            if google_error and _is_google_api_config_error(google_error):
                return f"{_google_config_message()}\nSerpAPI 备用搜索也失败: {serpapi_error}"
            return f"SerpAPI 搜索失败: {serpapi_error}"

    if google_error and _is_google_api_config_error(google_error):
        return _google_config_message()
    return "未配置联网搜索: 请配置 GOOGLE_API_KEY + GOOGLE_CSE_ID，或配置 SERPAPI_API_KEY。"


@tool
def read_website(url: str) -> str:
    """Read a web page when the user provides a URL to summarize, analyze, or extract."""
    first_error = None
    try:
        if _proxy_url():
            try:
                return _load_website(url, use_proxy=True)
            except Exception as exc:
                first_error = _safe_error(exc)
        return _load_website(url, use_proxy=False)
    except Exception as exc:
        final_error = _safe_error(exc)
        if first_error:
            return f"读取网页失败: 代理访问失败: {first_error}; 直连访问失败: {final_error}"
        return f"读取网页失败: {final_error}"
