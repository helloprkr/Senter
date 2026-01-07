#!/usr/bin/env python3
"""
Parallel Inference System for Senter
Executes response generation and research in parallel for faster, richer responses
"""

import logging
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional, Callable

logger = logging.getLogger("senter.parallel")

# Try to import dependencies
try:
    from web_search import search_web, search_news
    WEB_SEARCH_AVAILABLE = True
except ImportError:
    try:
        from Functions.web_search import search_web, search_news
        WEB_SEARCH_AVAILABLE = True
    except ImportError:
        WEB_SEARCH_AVAILABLE = False

try:
    from memory import ConversationMemory
    MEMORY_AVAILABLE = True
except ImportError:
    try:
        from Functions.memory import ConversationMemory
        MEMORY_AVAILABLE = True
    except ImportError:
        MEMORY_AVAILABLE = False

try:
    import requests
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass
class ResearchResult:
    """Results from parallel research"""
    web_results: list[dict] = field(default_factory=list)
    memory_results: list[dict] = field(default_factory=list)
    news_results: list[dict] = field(default_factory=list)
    source: str = ""


@dataclass
class ParallelResponse:
    """Combined response from parallel inference"""
    main_response: str = ""
    research: Optional[ResearchResult] = None
    enhanced_response: str = ""
    parallel_used: bool = False
    timing_ms: dict = field(default_factory=dict)


def needs_research(query: str) -> bool:
    """
    Determine if a query would benefit from parallel research.

    Args:
        query: User query

    Returns:
        True if research would likely improve the response
    """
    query_lower = query.lower()

    # Research-worthy patterns
    research_patterns = [
        r"\blatest\b", r"\bcurrent\b", r"\brecent\b", r"\btoday\b",
        r"\bnews\b", r"\bhappening\b", r"\bupdate\b",
        r"\bwhat is\b", r"\bwho is\b", r"\bwhere is\b",
        r"\bhow does\b", r"\bwhy does\b",
        r"\bprice of\b", r"\bcost of\b",
        r"\bstats\b", r"\bstatistics\b", r"\bdata on\b",
        r"\bresearch\b", r"\bstudy\b", r"\bfindings\b",
        r"\bcompare\b", r"\bvs\b", r"\bdifference between\b",
        r"\bbest\b.*\b(for|to)\b", r"\btop\b.*\b\d+\b",
    ]

    # Simple/conversational queries that don't need research
    simple_patterns = [
        r"^hi\b", r"^hello\b", r"^hey\b", r"^thanks\b",
        r"^help me (write|code|create)", r"^can you\b",
        r"\bremember when\b", r"\blast time\b",  # Memory queries, not web
        r"^(debug|fix|refactor)\b",  # Coding tasks
    ]

    # Check for simple patterns first
    for pattern in simple_patterns:
        if re.search(pattern, query_lower):
            return False

    # Check for research patterns
    for pattern in research_patterns:
        if re.search(pattern, query_lower):
            return True

    # Check query length - short queries likely don't need research
    if len(query.split()) < 4:
        return False

    return False


def _do_web_search(query: str) -> list[dict]:
    """Execute web search"""
    if not WEB_SEARCH_AVAILABLE:
        return []

    try:
        results = search_web(query, max_results=3)
        return results
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
        return []


def _do_news_search(query: str) -> list[dict]:
    """Execute news search"""
    if not WEB_SEARCH_AVAILABLE:
        return []

    try:
        results = search_news(query, max_results=3)
        return results
    except Exception as e:
        logger.warning(f"News search failed: {e}")
        return []


def _do_memory_search(query: str, senter_root: Path) -> list[dict]:
    """Execute memory search"""
    if not MEMORY_AVAILABLE:
        return []

    try:
        memory = ConversationMemory(senter_root)
        chunks = memory.search_memory(query, limit=3)
        return [
            {
                "content": chunk.content,
                "date": chunk.timestamp[:10] if chunk.timestamp else "",
                "role": chunk.role
            }
            for chunk in chunks
        ]
    except Exception as e:
        logger.warning(f"Memory search failed: {e}")
        return []


def _generate_response(query: str, model: str = "llama3.2") -> str:
    """Generate LLM response"""
    if not OLLAMA_AVAILABLE:
        return ""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": query,
                "stream": False,
                "options": {"temperature": 0.7}
            },
            timeout=60
        )

        if response.status_code == 200:
            return response.json().get("response", "")
        return ""
    except Exception as e:
        logger.error(f"Response generation failed: {e}")
        return ""


def _format_research_context(research: ResearchResult) -> str:
    """Format research results for inclusion in prompt"""
    parts = []

    if research.web_results:
        parts.append("[Web Search Results:]")
        for r in research.web_results[:3]:
            parts.append(f"- {r.get('title', 'No title')}")
            snippet = r.get('snippet', '')[:200]
            if snippet:
                parts.append(f"  {snippet}")

    if research.news_results:
        parts.append("\n[Recent News:]")
        for r in research.news_results[:2]:
            parts.append(f"- {r.get('title', 'No title')} ({r.get('date', 'recent')})")

    if research.memory_results:
        parts.append("\n[From Past Conversations:]")
        for r in research.memory_results[:2]:
            parts.append(f"- [{r.get('date', '?')}] {r.get('content', '')[:100]}...")

    return "\n".join(parts)


def parallel_respond(
    query: str,
    senter_root: Path = None,
    model: str = "llama3.2",
    include_news: bool = True
) -> ParallelResponse:
    """
    Generate a response with parallel research.

    Args:
        query: User query
        senter_root: Senter root path
        model: LLM model to use
        include_news: Whether to include news search

    Returns:
        ParallelResponse with main response and research results
    """
    import time
    start_time = time.time()

    result = ParallelResponse()
    senter_root = senter_root or Path(".")

    # Check if research is needed
    if not needs_research(query):
        # Simple query - just generate response
        result.main_response = _generate_response(query, model)
        result.enhanced_response = result.main_response
        result.parallel_used = False
        result.timing_ms["total"] = int((time.time() - start_time) * 1000)
        return result

    result.parallel_used = True
    research = ResearchResult()

    # Run tasks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}

        # Submit all research tasks
        futures["web"] = executor.submit(_do_web_search, query)
        futures["memory"] = executor.submit(_do_memory_search, query, senter_root)
        if include_news:
            futures["news"] = executor.submit(_do_news_search, query)

        # Collect results as they complete
        for name, future in futures.items():
            try:
                res = future.result(timeout=10)
                if name == "web":
                    research.web_results = res
                elif name == "memory":
                    research.memory_results = res
                elif name == "news":
                    research.news_results = res
                result.timing_ms[name] = int((time.time() - start_time) * 1000)
            except Exception as e:
                logger.warning(f"{name} task failed: {e}")

    # Build enhanced prompt with research context
    research_context = _format_research_context(research)

    if research_context:
        enhanced_prompt = f"""Based on the following research context, answer the user's question.

{research_context}

User Question: {query}

Provide a helpful response that incorporates relevant information from the research above."""

        result.main_response = _generate_response(enhanced_prompt, model)
        result.research = research
        research.source = "parallel"
    else:
        # No research results, generate normal response
        result.main_response = _generate_response(query, model)

    result.enhanced_response = result.main_response
    result.timing_ms["total"] = int((time.time() - start_time) * 1000)

    logger.info(f"Parallel inference completed in {result.timing_ms['total']}ms (parallel={result.parallel_used})")

    return result


def parallel_respond_streaming(
    query: str,
    on_chunk: Callable[[str], None],
    senter_root: Path = None,
    model: str = "llama3.2"
) -> ParallelResponse:
    """
    Generate a streaming response with parallel research.
    Research runs while initial response streams.

    Args:
        query: User query
        on_chunk: Callback for each response chunk
        senter_root: Senter root path
        model: LLM model to use

    Returns:
        ParallelResponse with research results (response delivered via callback)
    """
    import time
    start_time = time.time()

    result = ParallelResponse()
    senter_root = senter_root or Path(".")

    # For simple queries, just stream response
    if not needs_research(query):
        result.main_response = _generate_response(query, model)
        on_chunk(result.main_response)
        result.enhanced_response = result.main_response
        result.parallel_used = False
        result.timing_ms["total"] = int((time.time() - start_time) * 1000)
        return result

    result.parallel_used = True
    research = ResearchResult()

    # Run research in parallel while streaming response
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Start research tasks
        web_future = executor.submit(_do_web_search, query)
        memory_future = executor.submit(_do_memory_search, query, senter_root)

        # Generate initial response
        initial_response = _generate_response(query, model)
        on_chunk(initial_response)
        result.main_response = initial_response

        # Collect research results
        try:
            research.web_results = web_future.result(timeout=5)
        except:
            pass

        try:
            research.memory_results = memory_future.result(timeout=5)
        except:
            pass

    # If we got research results, add an addendum
    research_context = _format_research_context(research)
    if research_context:
        addendum = f"\n\n---\n**Additional Context Found:**\n{research_context}"
        on_chunk(addendum)
        result.research = research
        result.enhanced_response = initial_response + addendum
    else:
        result.enhanced_response = initial_response

    result.timing_ms["total"] = int((time.time() - start_time) * 1000)
    return result


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Senter Parallel Inference")
    parser.add_argument("query", nargs="?", help="Query to process")
    parser.add_argument("--check", "-c", help="Check if query needs research")
    parser.add_argument("--test", action="store_true", help="Run test queries")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.check:
        needs = needs_research(args.check)
        print(f"\nQuery: \"{args.check}\"")
        print(f"Needs research: {needs}")

    elif args.test:
        print("\nTesting needs_research() heuristic...\n")

        test_queries = [
            ("Hello, how are you?", False),
            ("What's the latest news on AI?", True),
            ("Help me write a Python function", False),
            ("What is the current price of Bitcoin?", True),
            ("Debug this code for me", False),
            ("What are the best practices for React in 2026?", True),
            ("Thanks!", False),
            ("Compare Python vs JavaScript for web development", True),
            ("Who won the Super Bowl?", True),
            ("Can you refactor this function?", False),
        ]

        passed = 0
        for query, expected in test_queries:
            actual = needs_research(query)
            status = "PASS" if actual == expected else "FAIL"
            if actual == expected:
                passed += 1
            print(f"[{status}] \"{query[:40]}...\"")
            print(f"       Expected: {expected}, Got: {actual}")

        print(f"\n{passed}/{len(test_queries)} tests passed")

    elif args.query:
        print(f"\nProcessing: \"{args.query}\"")
        print("=" * 50)

        result = parallel_respond(args.query, Path("."))

        print(f"\nParallel used: {result.parallel_used}")
        print(f"Timing: {result.timing_ms}")

        if result.research:
            print(f"\nResearch found:")
            print(f"  Web results: {len(result.research.web_results)}")
            print(f"  Memory results: {len(result.research.memory_results)}")
            print(f"  News results: {len(result.research.news_results)}")

        print(f"\nResponse:\n{result.enhanced_response[:500]}...")

    else:
        parser.print_help()
