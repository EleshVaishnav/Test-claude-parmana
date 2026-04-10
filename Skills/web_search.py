# ╔══════════════════════════════════════════════════════════════════╗
# ║           PARMANA 2.0 — Skill: Web Search                       ║
# ║  DuckDuckGo search. No API key required.                        ║
# ║  Self-registers into the global registry on import.             ║
# ╚══════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import logging
from typing import Optional

import yaml
from duckduckgo_search import DDGS

from Skills.registry import Skill, SkillParam, registry

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

def _load_cfg() -> dict:
    try:
        with open("config.yaml") as f:
            return yaml.safe_load(f).get("skills", {}).get("web_search", {})
    except Exception:
        return {}

_cfg = _load_cfg()
_MAX_RESULTS = _cfg.get("max_results", 8)
_TIMEOUT     = _cfg.get("timeout", 10)
_ENABLED     = _cfg.get("enabled", True)


# ── Handler ───────────────────────────────────────────────────────────────────

async def web_search(
    query: str,
    max_results: int = _MAX_RESULTS,
    region: str = "wt-wt",
    safe_search: str = "moderate",
) -> str:
    """
    Search the web via DuckDuckGo and return formatted results.

    Args:
        query:       Search query string.
        max_results: Number of results to return (default from config).
        region:      DDG region code. "wt-wt" = worldwide.
        safe_search: "on" | "moderate" | "off"

    Returns:
        Formatted string of results, one per line:
        [1] Title — URL\nSnippet\n
    """
    if not query or not query.strip():
        return "Error: empty query."

    logger.debug(f"web_search: query='{query}' max={max_results}")

    try:
        with DDGS(timeout=_TIMEOUT) as ddgs:
            raw = list(
                ddgs.text(
                    query,
                    region=region,
                    safesearch=safe_search,
                    max_results=max_results,
                )
            )
    except Exception as e:
        logger.warning(f"DDG search failed: {e}")
        return f"Search failed: {e}"

    if not raw:
        return f"No results found for: {query}"

    lines = []
    for i, r in enumerate(raw, 1):
        title   = r.get("title", "").strip()
        href    = r.get("href", "").strip()
        snippet = r.get("body", "").strip()
        lines.append(f"[{i}] {title} — {href}\n    {snippet}")

    return "\n\n".join(lines)


async def web_news(
    query: str,
    max_results: int = 5,
    region: str = "wt-wt",
) -> str:
    """
    Search DuckDuckGo News specifically. Good for recent events.

    Args:
        query:       News search query.
        max_results: Number of articles to return.
        region:      DDG region code.

    Returns:
        Formatted news results with date and source.
    """
    if not query or not query.strip():
        return "Error: empty query."

    logger.debug(f"web_news: query='{query}' max={max_results}")

    try:
        with DDGS(timeout=_TIMEOUT) as ddgs:
            raw = list(
                ddgs.news(
                    query,
                    region=region,
                    max_results=max_results,
                )
            )
    except Exception as e:
        logger.warning(f"DDG news search failed: {e}")
        return f"News search failed: {e}"

    if not raw:
        return f"No news found for: {query}"

    lines = []
    for i, r in enumerate(raw, 1):
        title  = r.get("title", "").strip()
        href   = r.get("url", "").strip()
        source = r.get("source", "").strip()
        date   = r.get("date", "").strip()
        body   = r.get("body", "").strip()
        lines.append(f"[{i}] {title}\n    {source} · {date} — {href}\n    {body}")

    return "\n\n".join(lines)


# ── Registration ──────────────────────────────────────────────────────────────

registry.register(Skill(
    name="web_search",
    description="Search the web via DuckDuckGo. Use for current events, docs, anything not in training data.",
    params=[
        SkillParam(
            name="query",
            type="string",
            description="Search query string.",
            required=True,
        ),
        SkillParam(
            name="max_results",
            type="number",
            description="Number of results to return.",
            required=False,
            default=_MAX_RESULTS,
        ),
        SkillParam(
            name="region",
            type="string",
            description="DDG region code. Default 'wt-wt' = worldwide.",
            required=False,
            default="wt-wt",
        ),
        SkillParam(
            name="safe_search",
            type="string",
            description="Safe search level: on | moderate | off",
            required=False,
            default="moderate",
            enum=["on", "moderate", "off"],
        ),
    ],
    handler=web_search,
    enabled=_ENABLED,
    tags=["search", "web", "realtime"],
    timeout=_TIMEOUT,
))

registry.register(Skill(
    name="web_news",
    description="Search DuckDuckGo News for recent articles. Better than web_search for current events.",
    params=[
        SkillParam(
            name="query",
            type="string",
            description="News search query.",
            required=True,
        ),
        SkillParam(
            name="max_results",
            type="number",
            description="Number of articles to return.",
            required=False,
            default=5,
        ),
        SkillParam(
            name="region",
            type="string",
            description="DDG region code.",
            required=False,
            default="wt-wt",
        ),
    ],
    handler=web_news,
    enabled=_ENABLED,
    tags=["search", "news", "realtime"],
    timeout=_TIMEOUT,
))

logger.debug("web_search + web_news skills registered.")
