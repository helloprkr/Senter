#!/usr/bin/env python3
"""
RE-002: Senter Actually Researches the Web

Multi-query web research with actual content fetching and extraction.

VALUE: Senter doesn't just ask the LLM—it searches the real web, reads
actual articles, and gathers information from multiple sources.
"""

import json
import logging
import time
import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger("senter.research.deep_researcher")

# Try to import required libraries
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    logger.warning("httpx not installed - run: pip install httpx")

try:
    from readability import Document
    READABILITY_AVAILABLE = True
except ImportError:
    READABILITY_AVAILABLE = False
    logger.warning("readability-lxml not installed - run: pip install readability-lxml")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logger.warning("beautifulsoup4 not installed - run: pip install beautifulsoup4")


@dataclass
class ResearchSource:
    """A source document with extracted content."""
    url: str
    title: str
    content: str  # Extracted readable text
    key_points: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    fetch_time_ms: int = 0
    domain: str = ""
    error: Optional[str] = None

    def __post_init__(self):
        if not self.domain and self.url:
            try:
                self.domain = urlparse(self.url).netloc
            except:
                pass

    def to_dict(self) -> Dict[str, Any]:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content[:500] + "..." if len(self.content) > 500 else self.content,
            "key_points": self.key_points,
            "relevance_score": self.relevance_score,
            "domain": self.domain,
            "error": self.error,
        }


class DeepResearcher:
    """
    Performs deep web research on a topic.

    Process:
    1. Generate diverse search queries (intro, examples, comparisons, pitfalls)
    2. Search DuckDuckGo for each query
    3. Fetch actual page content
    4. Extract readable text using readability
    5. Score relevance and extract key points
    """

    # Query templates for diverse searches
    QUERY_TEMPLATES = [
        "{topic}",                              # Direct search
        "{topic} tutorial introduction",         # Beginner angle
        "{topic} examples use cases",           # Practical examples
        "{topic} vs alternatives comparison",   # Comparisons
        "{topic} best practices pitfalls",      # Gotchas
    ]

    # User agents for requests
    USER_AGENTS = [
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ]

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        max_sources: int = 8,
        fetch_timeout: float = 10.0,
        max_concurrent_fetches: int = 4
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.max_sources = max_sources
        self.fetch_timeout = fetch_timeout
        self.max_concurrent_fetches = max_concurrent_fetches

        # Import web search from existing module
        self._search_web = None
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent / "Functions"))
            from web_search import search_web
            self._search_web = search_web
        except ImportError as e:
            logger.warning(f"Could not import web_search: {e}")

    def research(self, topic: str) -> List[ResearchSource]:
        """
        Perform deep research on a topic.

        Args:
            topic: The topic to research

        Returns:
            List of ResearchSource objects with content and key points
        """
        logger.info(f"Starting deep research on: {topic}")
        start_time = time.time()

        # Step 1: Generate search queries
        queries = self._generate_queries(topic)
        logger.info(f"Generated {len(queries)} search queries")

        # Step 2: Search and collect URLs
        urls = self._collect_urls(queries)
        logger.info(f"Collected {len(urls)} unique URLs")

        if not urls:
            logger.warning("No URLs found")
            return []

        # Step 3: Fetch and extract content
        sources = self._fetch_sources(urls[:self.max_sources * 2])  # Fetch extra in case some fail
        logger.info(f"Fetched {len(sources)} sources")

        # Step 4: Score relevance and extract key points
        sources = self._process_sources(sources, topic)

        # Step 5: Filter and sort
        valid_sources = [s for s in sources if not s.error and s.content]
        valid_sources.sort(key=lambda s: s.relevance_score, reverse=True)
        valid_sources = valid_sources[:self.max_sources]

        elapsed = time.time() - start_time
        logger.info(f"Deep research complete: {len(valid_sources)} sources in {elapsed:.1f}s")

        return valid_sources

    def _generate_queries(self, topic: str) -> List[str]:
        """Generate diverse search queries for the topic."""
        queries = []
        for template in self.QUERY_TEMPLATES:
            query = template.format(topic=topic)
            queries.append(query)
        return queries

    def _collect_urls(self, queries: List[str]) -> List[str]:
        """Search and collect unique URLs."""
        if not self._search_web:
            logger.error("Web search not available")
            return []

        seen_urls = set()
        urls = []

        for query in queries:
            try:
                results = self._search_web(query, max_results=5)
                for result in results:
                    url = result.get("url", "")
                    if url and url not in seen_urls:
                        # Skip certain domains
                        domain = urlparse(url).netloc.lower()
                        if any(skip in domain for skip in ["youtube.com", "facebook.com", "twitter.com", "instagram.com"]):
                            continue
                        seen_urls.add(url)
                        urls.append(url)
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")

        return urls

    def _fetch_sources(self, urls: List[str]) -> List[ResearchSource]:
        """Fetch content from URLs concurrently."""
        if not HTTPX_AVAILABLE:
            logger.error("httpx not available, cannot fetch content")
            return []

        sources = []

        with ThreadPoolExecutor(max_workers=self.max_concurrent_fetches) as executor:
            future_to_url = {
                executor.submit(self._fetch_single, url): url
                for url in urls
            }

            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    source = future.result()
                    if source:
                        sources.append(source)
                except Exception as e:
                    logger.debug(f"Fetch failed for {url}: {e}")

        return sources

    def _fetch_single(self, url: str) -> Optional[ResearchSource]:
        """Fetch and extract content from a single URL."""
        start_time = time.time()

        try:
            # Rotate user agent
            ua_index = hash(url) % len(self.USER_AGENTS)
            headers = {"User-Agent": self.USER_AGENTS[ua_index]}

            with httpx.Client(timeout=self.fetch_timeout, follow_redirects=True) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()

                html = response.text
                fetch_time = int((time.time() - start_time) * 1000)

                # Extract readable content
                title, content = self._extract_content(html)

                if not content or len(content) < 100:
                    return None

                return ResearchSource(
                    url=url,
                    title=title or "Untitled",
                    content=content,
                    fetch_time_ms=fetch_time
                )

        except Exception as e:
            logger.debug(f"Failed to fetch {url}: {e}")
            return ResearchSource(
                url=url,
                title="",
                content="",
                error=str(e)
            )

    def _extract_content(self, html: str) -> tuple[str, str]:
        """Extract title and readable content from HTML."""
        title = ""
        content = ""

        # Try readability first (best quality)
        if READABILITY_AVAILABLE:
            try:
                doc = Document(html)
                title = doc.title()
                content = doc.summary()

                # Clean up HTML from summary
                if BS4_AVAILABLE:
                    soup = BeautifulSoup(content, "html.parser")
                    content = soup.get_text(separator="\n", strip=True)
                else:
                    # Basic HTML stripping
                    content = re.sub(r'<[^>]+>', '', content)
                    content = re.sub(r'\s+', ' ', content).strip()

            except Exception as e:
                logger.debug(f"Readability extraction failed: {e}")

        # Fallback to basic extraction
        if not content and BS4_AVAILABLE:
            try:
                soup = BeautifulSoup(html, "html.parser")

                # Get title
                title_tag = soup.find("title")
                if title_tag:
                    title = title_tag.get_text().strip()

                # Remove script and style elements
                for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
                    element.decompose()

                # Get main content areas
                main = soup.find("main") or soup.find("article") or soup.find(class_=re.compile("content|article|post"))
                if main:
                    content = main.get_text(separator="\n", strip=True)
                else:
                    content = soup.get_text(separator="\n", strip=True)

            except Exception as e:
                logger.debug(f"BeautifulSoup extraction failed: {e}")

        # Clean up content
        if content:
            # Remove excessive whitespace
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            content = "\n".join(lines)

            # Truncate very long content
            if len(content) > 10000:
                content = content[:10000] + "..."

        return title, content

    def _process_sources(self, sources: List[ResearchSource], topic: str) -> List[ResearchSource]:
        """Score relevance and extract key points from sources."""
        topic_words = set(topic.lower().split())

        for source in sources:
            if source.error or not source.content:
                continue

            # Calculate relevance score
            content_lower = source.content.lower()
            title_lower = source.title.lower()

            # Word overlap scoring
            content_words = set(content_lower.split())
            word_overlap = len(topic_words & content_words) / max(len(topic_words), 1)

            # Title match bonus
            title_match = sum(1 for w in topic_words if w in title_lower) / max(len(topic_words), 1)

            # Content length score (prefer substantial content)
            length_score = min(len(source.content) / 5000, 1.0)

            # Calculate final score
            source.relevance_score = round(
                0.4 * word_overlap +
                0.3 * title_match +
                0.3 * length_score,
                2
            )

            # Extract key points using LLM (optional, can be slow)
            # For now, extract first few meaningful sentences
            source.key_points = self._extract_key_sentences(source.content, topic)

        return sources

    def _extract_key_sentences(self, content: str, topic: str, max_points: int = 3) -> List[str]:
        """Extract key sentences that relate to the topic."""
        topic_words = set(topic.lower().split())
        sentences = re.split(r'[.!?]+', content)

        scored_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) < 30 or len(sent) > 300:
                continue

            sent_lower = sent.lower()
            # Score by topic word presence
            score = sum(1 for w in topic_words if w in sent_lower)

            if score > 0:
                scored_sentences.append((score, sent))

        # Sort by score and return top N
        scored_sentences.sort(key=lambda x: x[0], reverse=True)
        return [sent for _, sent in scored_sentences[:max_points]]


# CLI for testing
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Test deep research")
    parser.add_argument("topic", nargs="?", default="event sourcing patterns", help="Topic to research")
    parser.add_argument("--max-sources", "-n", type=int, default=5, help="Max sources to fetch")

    args = parser.parse_args()

    researcher = DeepResearcher(max_sources=args.max_sources)

    print(f"\n{'='*60}")
    print(f"Deep Research: {args.topic}")
    print(f"{'='*60}")

    sources = researcher.research(args.topic)

    if not sources:
        print("\nNo sources found. Check:")
        print("  - Is duckduckgo-search installed?")
        print("  - Is httpx installed?")
        print("  - Is internet connected?")
    else:
        print(f"\nFound {len(sources)} sources:\n")
        for i, source in enumerate(sources, 1):
            print(f"{i}. {source.title}")
            print(f"   URL: {source.url}")
            print(f"   Domain: {source.domain}")
            print(f"   Relevance: {source.relevance_score}")
            print(f"   Content length: {len(source.content)} chars")
            if source.key_points:
                print(f"   Key points:")
                for point in source.key_points:
                    print(f"     - {point[:100]}...")
            print()

    print(f"{'='*60}")
