"""
app/tools/tavily_provider.py
=============================
Tavily search provider implementation.

Tavily is an AI-optimised search API designed specifically for LLM applications.
It returns clean, relevant results with content snippets without requiring
HTML parsing or scraping logic.

API reference: https://docs.tavily.com/docs/python-sdk/tavily-search/api-reference

Authentication:
    Requires TAVILY_API_KEY in the environment.
    Get a free API key at https://tavily.com — free tier includes 1000 searches/month.

Search depths:
    "basic"    — faster, cheaper, suitable for most queries (~1–2 API credits)
    "advanced" — deeper crawl, higher quality for complex queries (~2–4 API credits)
    Controlled by TAVILY_SEARCH_DEPTH env var (default: "basic")

Web fetch:
    Uses Tavily's Extract API to retrieve clean page content.
    Falls back to urllib if the Extract API fails or returns no content.

Limitations:
    - No rate limit handling / retry logic in MVP ⚠️
    - extract() API may not be available on free tier — falls back to urllib ⚠️
"""

from __future__ import annotations
import urllib.request
import urllib.error
from typing import Optional

from tavily import TavilyClient

from app.tools.base import SearchProvider, SearchResult
from app.config import TAVILY_API_KEY, TAVILY_SEARCH_DEPTH


class TavilySearchProvider(SearchProvider):
    """
    SearchProvider implementation backed by the Tavily API.

    Instantiated once as a singleton by the factory and reused across
    all sub-agent calls within a server session.
    """

    def __init__(self):
        if not TAVILY_API_KEY:
            raise EnvironmentError(
                "TAVILY_API_KEY is not set. "
                "Add it to your .env file. Get a key at https://tavily.com"
            )
        self._client = TavilyClient(api_key=TAVILY_API_KEY)
        # "basic" is cheaper; switch to "advanced" for higher-quality deep research
        self._search_depth = TAVILY_SEARCH_DEPTH

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        Execute a Tavily web search and return normalised SearchResult objects.

        Args:
            query:       Search query string.
            max_results: Number of results to request (Tavily max is 20).

        Returns:
            list[SearchResult]: Ranked results, most relevant first.
                                Returns empty list on any API error.
        """
        try:
            response = self._client.search(
                query=query,
                search_depth=self._search_depth,
                max_results=min(max_results, 20),  # Tavily hard cap is 20
                include_answer=False,               # raw results only, no Tavily summary
                include_raw_content=False,          # snippets only; fetch() for full content
            )
            return [
                SearchResult(
                    url=r.get("url", ""),
                    title=r.get("title", r.get("url", "Unknown")),
                    content=r.get("content", ""),
                    published_date=r.get("published_date"),
                    score=r.get("score"),
                )
                for r in response.get("results", [])
                if r.get("url")  # skip results with no URL
            ]
        except Exception as exc:
            # Return empty rather than crashing the sub-agent — caller handles empty results
            return []

    def fetch(self, url: str) -> str:
        """
        Fetch the full text content of a URL.

        Tries Tavily Extract API first (cleaner content).
        Falls back to urllib if Extract fails or returns nothing.

        Args:
            url: Full URL to fetch.

        Returns:
            str: Extracted page text. Empty string on failure.
        """
        # Try Tavily Extract for clean content
        try:
            response = self._client.extract(urls=[url])
            results = response.get("results", [])
            if results and results[0].get("raw_content"):
                return results[0]["raw_content"][:10000]  # cap at 10k chars
        except Exception:
            pass  # fall through to urllib fallback

        # Fallback: plain HTTP fetch via urllib (no JS rendering)
        return _urllib_fetch(url)


# ── Module-level helper ───────────────────────────────────────────────────────

def _urllib_fetch(url: str, timeout: int = 10) -> str:
    """
    Fetch a URL using urllib as a fallback when the provider's fetch API fails.

    Returns raw HTML/text — not cleaned. Sub-agent will extract relevant content.
    Caps output at 10,000 characters to avoid overwhelming Claude's context.

    Args:
        url:     Full URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        str: Raw page content (up to 10,000 chars). Empty string on error.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Research Bot; +https://github.com)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read().decode("utf-8", errors="replace")
            return content[:10000]
    except (urllib.error.URLError, Exception):
        return ""
