"""
Web Search - Search the web via DuckDuckGo.

Provides web search capability without API keys.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import httpx


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str


class WebSearch:
    """
    Web search capability using DuckDuckGo.

    No API key required - uses the instant answer API.
    """

    BASE_URL = "https://api.duckduckgo.com/"

    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(10.0, connect=5.0),
                follow_redirects=True,
            )
        return self._client

    async def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> List[SearchResult]:
        """
        Search the web.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            List of search results
        """
        client = await self._get_client()

        # DuckDuckGo Instant Answer API
        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
        }

        try:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            results = []

            # Abstract (main answer)
            if data.get("Abstract"):
                results.append(
                    SearchResult(
                        title=data.get("Heading", "Result"),
                        url=data.get("AbstractURL", ""),
                        snippet=data.get("Abstract", ""),
                    )
                )

            # Related topics
            for topic in data.get("RelatedTopics", [])[:max_results]:
                if isinstance(topic, dict):
                    if "Text" in topic:
                        results.append(
                            SearchResult(
                                title=topic.get("FirstURL", "").split("/")[-1].replace("_", " "),
                                url=topic.get("FirstURL", ""),
                                snippet=topic.get("Text", ""),
                            )
                        )

            # Results section
            for result in data.get("Results", [])[:max_results]:
                if isinstance(result, dict):
                    results.append(
                        SearchResult(
                            title=result.get("Text", "")[:50],
                            url=result.get("FirstURL", ""),
                            snippet=result.get("Text", ""),
                        )
                    )

            return results[:max_results]

        except httpx.HTTPError:
            return []
        except Exception:
            return []

    async def get_instant_answer(self, query: str) -> Optional[str]:
        """
        Get instant answer for a query.

        Args:
            query: The query

        Returns:
            Instant answer text or None
        """
        client = await self._get_client()

        params = {
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1",
        }

        try:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            # Check various answer types
            if data.get("Answer"):
                return data["Answer"]
            if data.get("Abstract"):
                return data["Abstract"]
            if data.get("Definition"):
                return data["Definition"]

            return None

        except Exception:
            return None

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None


# Convenience function
async def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Search the web (convenience function).

    Args:
        query: Search query
        max_results: Maximum results

    Returns:
        List of result dictionaries
    """
    searcher = WebSearch()
    try:
        results = await searcher.search(query, max_results)
        return [
            {"title": r.title, "url": r.url, "snippet": r.snippet}
            for r in results
        ]
    finally:
        await searcher.close()
