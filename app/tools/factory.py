"""
app/tools/factory.py
=====================
Factory function for instantiating the configured search provider.

The active provider is selected via the SEARCH_PROVIDER environment variable.
Defaults to "tavily" if not set.

To add a new provider:
    1. Create app/tools/{name}_provider.py implementing SearchProvider
    2. Import it here and add it to _PROVIDERS
    3. Set SEARCH_PROVIDER={name} in .env

Currently supported providers:
    tavily      — Tavily AI search API (https://tavily.com)

Planned / community-contributed:
    bing        — Microsoft Bing Search API
    serpapi     — SerpAPI (Google, Bing, DuckDuckGo via one API)
    brave       — Brave Search API
    duckduckgo  — DuckDuckGo (free, no API key)

Usage:
    from app.tools.factory import get_search_provider
    provider = get_search_provider()          # uses SEARCH_PROVIDER env var
    results = provider.search("AI in 2025")
"""

from __future__ import annotations

from app.tools.base import SearchProvider
from app.config import SEARCH_PROVIDER


# Registry of available providers — add new ones here
_PROVIDERS: dict[str, str] = {
    "tavily": "app.tools.tavily_provider.TavilySearchProvider",
}


def get_search_provider() -> SearchProvider:
    """
    Instantiate and return the search provider configured by SEARCH_PROVIDER env var.

    Performs lazy import so only the selected provider's dependencies are loaded.
    This means you don't need Tavily installed if you're using Bing, for example.

    Returns:
        SearchProvider: Ready-to-use provider instance.

    Raises:
        ValueError:         SEARCH_PROVIDER names an unsupported provider.
        EnvironmentError:   Required API key for the provider is missing.
        ImportError:        Provider's package dependency is not installed.
    """
    provider_name = SEARCH_PROVIDER.lower().strip()

    if provider_name not in _PROVIDERS:
        supported = ", ".join(_PROVIDERS.keys())
        raise ValueError(
            f"Unsupported SEARCH_PROVIDER='{provider_name}'. "
            f"Supported providers: {supported}"
        )

    # Lazy import — only load the provider module that's actually needed
    module_path, class_name = _PROVIDERS[provider_name].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    provider_class = getattr(module, class_name)

    return provider_class()
