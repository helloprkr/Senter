#!/usr/bin/env python3
"""
Web Search Module for Senter
Uses duckduckgo-search package for reliable web search
"""

import sys
from pathlib import Path
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger("senter.web_search")

try:
    from duckduckgo_search import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("duckduckgo-search not installed - run: pip install duckduckgo-search")


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Perform web search using DuckDuckGo.

    Args:
        query: Search query string
        max_results: Maximum number of results to return (default 5)

    Returns:
        List of dicts with 'title', 'url', 'snippet' keys
    """
    if not DDGS_AVAILABLE:
        logger.error("duckduckgo-search not available")
        return []

    if not query or not query.strip():
        return []

    logger.info(f"Web search: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                }
                for r in results
            ]
            logger.info(f"Found {len(formatted)} results")
            return formatted
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []


def search_news(query: str, max_results: int = 5) -> list[dict]:
    """
    Search recent news articles.

    Args:
        query: News search query
        max_results: Maximum results to return

    Returns:
        List of news articles with title, url, snippet, date, source
    """
    if not DDGS_AVAILABLE:
        logger.error("duckduckgo-search not available")
        return []

    if not query or not query.strip():
        return []

    logger.info(f"News search: {query}")

    try:
        with DDGS() as ddgs:
            results = list(ddgs.news(query, max_results=max_results))
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "snippet": r.get("body", ""),
                    "date": r.get("date", ""),
                    "source": r.get("source", "")
                }
                for r in results
            ]
            logger.info(f"Found {len(formatted)} news articles")
            return formatted
    except Exception as e:
        logger.error(f"News search error: {e}")
        return []


def search_images(query: str, max_results: int = 5) -> list[dict]:
    """
    Search for images.

    Args:
        query: Image search query
        max_results: Maximum results to return

    Returns:
        List of images with title, url, image_url, thumbnail
    """
    if not DDGS_AVAILABLE:
        return []

    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
            formatted = [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "image_url": r.get("image", ""),
                    "thumbnail": r.get("thumbnail", "")
                }
                for r in results
            ]
            return formatted
    except Exception as e:
        logger.error(f"Image search error: {e}")
        return []


def get_instant_answer(query: str) -> Optional[dict]:
    """
    Get instant answer for factual queries.

    Args:
        query: Question or factual query

    Returns:
        Dict with 'answer', 'source', 'url' or None
    """
    if not DDGS_AVAILABLE:
        return None

    try:
        with DDGS() as ddgs:
            # Use answers endpoint for instant answers
            results = list(ddgs.answers(query))
            if results:
                r = results[0]
                return {
                    "answer": r.get("text", ""),
                    "source": r.get("source", ""),
                    "url": r.get("url", "")
                }
            return None
    except Exception as e:
        logger.error(f"Instant answer error: {e}")
        return None


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Web Search for Senter")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of results")
    parser.add_argument("--news", action="store_true", help="Search news instead")
    parser.add_argument("--images", action="store_true", help="Search images instead")
    parser.add_argument("--answer", action="store_true", help="Get instant answer")

    args = parser.parse_args()

    print(f"\n{'='*60}")

    if args.answer:
        print(f"Instant Answer: '{args.query}'")
        print("="*60)
        result = get_instant_answer(args.query)
        if result:
            print(f"\nAnswer: {result['answer']}")
            print(f"Source: {result['source']}")
            print(f"URL: {result['url']}")
        else:
            print("No instant answer found")

    elif args.news:
        print(f"News Search: '{args.query}'")
        print("="*60)
        results = search_news(args.query, args.count)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['title']}")
            print(f"   Date: {r['date']}")
            print(f"   Source: {r['source']}")
            print(f"   URL: {r['url']}")

    elif args.images:
        print(f"Image Search: '{args.query}'")
        print("="*60)
        results = search_images(args.query, args.count)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['title']}")
            print(f"   URL: {r['image_url']}")

    else:
        print(f"Web Search: '{args.query}'")
        print("="*60)
        results = search_web(args.query, args.count)
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r['title']}")
            print(f"   URL: {r['url']}")
            print(f"   {r['snippet'][:150]}..." if len(r['snippet']) > 150 else f"   {r['snippet']}")

    print(f"\n{'='*60}\n")
