import html
import re
from urllib.parse import urlparse

import requests

from app import config
from app.data import ScoredDoc


DUCKDUCKGO_HTML = "https://duckduckgo.com/html/"


def _trusted(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    for domain in config.TRUSTED_WEB_DOMAINS:
        domain = domain.lower()
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False


def _extract_results(page: str) -> list[tuple[str, str]]:
    pairs = re.findall(r'<a rel="nofollow" class="result__a" href="(.*?)">(.*?)</a>', page)
    cleaned = []
    for url, title in pairs:
        title_text = re.sub(r"<.*?>", "", title)
        cleaned.append((html.unescape(url), html.unescape(title_text)))
    return cleaned


def retrieve_web_docs(query: str) -> list[ScoredDoc]:
    if not config.ENABLE_WEB_AUGMENTATION:
        return []

    try:
        resp = requests.post(
            DUCKDUCKGO_HTML,
            data={"q": query},
            timeout=config.WEB_SEARCH_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except Exception:
        return []

    docs: list[ScoredDoc] = []
    for url, title in _extract_results(resp.text):
        if not _trusted(url):
            continue

        try:
            page = requests.get(url, timeout=config.WEB_SEARCH_TIMEOUT_SECONDS)
            text = re.sub(r"\s+", " ", page.text)
            snippet = text[:1200]
        except Exception:
            continue

        docs.append(
            ScoredDoc(
                content=f"{title}\n\n{snippet}",
                doc=None,
                source=url,
                source_kind="web",
                doc_id=f"web::{abs(hash(url))}",
                metadata={"url": url, "title": title},
            )
        )
        if len(docs) >= config.WEB_TOP_K:
            break

    return docs
