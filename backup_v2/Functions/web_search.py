#!/usr/bin/env python3
"""
Web Search Module for Senter
Uses DuckDuckGo API for web search integration
"""

import sys
from pathlib import Path
import logging

# Add Senter to path
senter_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(senter_root))
sys.path.insert(1, str(senter_root / "Functions"))
sys.path.insert(2, str(senter_root / "Focuses"))

# Configure logging
logger = logging.getLogger("senter.web_search")

try:
    import requests
except ImportError:
    requests = None
    logger.warning("requests library not installed - web search will be limited")


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Perform web search using DuckDuckGo API
    
    Args:
        query: Search query
        max_results: Maximum number of results to return
        
    Returns:
        List of search results with title, url, snippet
    """
    if not requests:
        logger.error("requests library not available")
        return []
    
    logger.info(f"Web search query: {query}")
    
    try:
        # DuckDuckGo Instant Answer API
        url = "https://api.duckduckgo.com/"
        params = {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 0
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        results = []
        if "RelatedTopics" in data:
            for topic in data["RelatedTopics"][:max_results]:
                results.append({
                    "title": topic.get("Text", ""),
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")
                })
        
        logger.info(f"Found {len(results)} results")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        return []
    except Exception as e:
        logger.error(f"Web search error: {e}")
        return []


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Web Search for Senter")
    parser.add_argument("query", help="Search query")
    parser.add_argument("--count", "-n", type=int, default=5, help="Number of results")
    
    args = parser.parse_args()
    
    if args.query:
        results = search_web(args.query, args.count)
        print(f"🌐 Web Search Results for '{args.query}':")
        print("=" * 60)
        for i, result in enumerate(results, 1):
            print(f"\n{i + 1}. {result['title']}")
            print(f"   URL: {result['url']}")
            if i < len(results):
                print(f"   {result['snippet'][:150]}...")
            print()
    else:
        print("Usage: python3 Functions/web_search.py \"search query\"")
