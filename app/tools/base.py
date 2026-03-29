"""
app/tools/base.py
=================
Abstract base classes and shared data models for the search provider abstraction layer.

Every search provider (Tavily, Bing, SerpAPI, etc.) must implement the
SearchProvider interface. This keeps sub_agent.py completely decoupled from
any specific search service — swapping providers requires only changing the
SEARCH_PROVIDER environment variable and ensuring the provider class exists.

Classes:
    SearchResult    — normalised single search result returned by any provider
    SearchProvider  — abstract base class all providers must implement

Design principles:
    - Providers are stateless — instantiated once via factory, reused across requests
    - All providers return the same SearchResult shape so sub_agent.py never needs
      to know which provider is active
    - web_fetch() is part of the interface so providers can use their own
      fetch mechanism (e.g. Tavily Extract API) or fall back to plain HTTP
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """
    Normalised search result returned by any SearchProvider implementation.

    All provider-specific response formats are mapped to this structure
    so the rest of the system never depends on provider-specific fields.

    Fields:
        url             Full URL of the source page
        title           Page title or URL if title unavailable
        content         Snippet or extracted body text
        published_date  ISO date string if available; None otherwise
        score           Provider relevance score (0.0–1.0); None if not provided
    """
    url: str
    title: str
    content: str
    published_date: Optional[str] = None
    score: Optional[float] = None


class SearchProvider(ABC):
    """
    Abstract base class for all web search providers.

    Any class implementing this interface can be used as the search backend
    for ResearchSubAgent without modifying any other part of the system.

    To add a new provider:
        1. Create app/tools/{provider_name}_provider.py
        2. Subclass SearchProvider and implement search() and fetch()
        3. Register it in app/tools/factory.py
        4. Set SEARCH_PROVIDER={provider_name} in .env
    """

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        """
        Execute a web search and return normalised results.

        Args:
            query:       Natural language or keyword search query.
            max_results: Maximum number of results to return.

        Returns:
            list[SearchResult]: Ranked results, most relevant first.
        """
        ...

    @abstractmethod
    def fetch(self, url: str) -> str:
        """
        Fetch and return the text content of a web page.

        Args:
            url: Full URL of the page to fetch.

        Returns:
            str: Extracted text content. Empty string on failure.
        """
        ...

    def format_search_results(self, results: list[SearchResult]) -> str:
        """
        Format a list of SearchResults into a readable string for Claude's context.

        This is a shared utility — providers do not need to override it.
        Sub-agent receives this formatted string as the tool_result content.

        Args:
            results: List of SearchResult objects from search().

        Returns:
            str: Human-readable formatted results with index, title, URL, snippet.
        """
        if not results:
            return "No results found."

        lines = []
        for i, r in enumerate(results, 1):
            date_str = f" ({r.published_date})" if r.published_date else ""
            lines.append(f"[{i}] {r.title}{date_str}")
            lines.append(f"    URL: {r.url}")
            lines.append(f"    {r.content[:300]}{'...' if len(r.content) > 300 else ''}")
            lines.append("")
        return "\n".join(lines)
