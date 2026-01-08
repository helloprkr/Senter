#!/usr/bin/env python3
"""
RE-003: Senter Synthesizes Into Useful Summary

Multi-source synthesis using LLM to create actionable research summaries.

VALUE: Senter doesn't just dump articles—it reads them and tells you
what matters, synthesized from multiple sources with citations.
"""

import json
import logging
import time
import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("senter.research.synthesizer")

# Import ResearchSource from deep_researcher
try:
    from .deep_researcher import ResearchSource
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from deep_researcher import ResearchSource


@dataclass
class SynthesizedResearch:
    """A synthesized research summary with citations."""
    topic: str
    summary: str  # Main synthesized summary
    key_insights: List[str] = field(default_factory=list)  # Bullet points
    sources_used: List[str] = field(default_factory=list)  # URLs cited
    confidence: float = 0.0  # How confident we are (0-1)
    synthesis_time_ms: int = 0
    raw_source_count: int = 0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "summary": self.summary,
            "key_insights": self.key_insights,
            "sources_used": self.sources_used,
            "confidence": self.confidence,
            "synthesis_time_ms": self.synthesis_time_ms,
            "raw_source_count": self.raw_source_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SynthesizedResearch":
        return cls(**data)

    def format_for_display(self) -> str:
        """Format for human-readable display."""
        lines = []
        lines.append(f"# {self.topic}")
        lines.append("")
        lines.append(self.summary)
        lines.append("")

        if self.key_insights:
            lines.append("## Key Insights")
            for insight in self.key_insights:
                lines.append(f"• {insight}")
            lines.append("")

        if self.sources_used:
            lines.append("## Sources")
            for i, url in enumerate(self.sources_used, 1):
                lines.append(f"{i}. {url}")

        return "\n".join(lines)


class ResearchSynthesizer:
    """
    Synthesizes multiple research sources into actionable summaries.

    Process:
    1. Take top N sources from research
    2. Combine relevant excerpts
    3. Use LLM to synthesize into coherent summary
    4. Extract key insights as bullet points
    5. Track confidence based on source agreement
    """

    SYNTHESIS_PROMPT = """You are a research assistant synthesizing information from multiple web sources.

TOPIC: {topic}

SOURCES:
{sources}

INSTRUCTIONS:
1. Read all sources carefully
2. Synthesize the key information into a clear, actionable summary
3. Focus on practical insights the user can apply
4. Note when sources agree or disagree
5. Be concise but comprehensive

OUTPUT FORMAT (JSON):
{{
  "summary": "2-3 paragraph synthesis of the topic, written conversationally",
  "key_insights": [
    "First key actionable insight",
    "Second key insight",
    "Third key insight"
  ],
  "confidence": 0.8
}}

The confidence score should reflect:
- 0.9+: Strong agreement across sources, well-established topic
- 0.7-0.9: General agreement with some nuance
- 0.5-0.7: Mixed perspectives or emerging topic
- <0.5: Conflicting information or sparse sources

RESPOND WITH VALID JSON ONLY:"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        max_sources: int = 4,
        max_content_per_source: int = 2000
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.max_sources = max_sources
        self.max_content_per_source = max_content_per_source

    def synthesize(
        self,
        topic: str,
        sources: List[ResearchSource]
    ) -> SynthesizedResearch:
        """
        Synthesize multiple sources into a coherent summary.

        Args:
            topic: The research topic
            sources: List of ResearchSource objects from DeepResearcher

        Returns:
            SynthesizedResearch object with summary and insights
        """
        logger.info(f"Synthesizing {len(sources)} sources for: {topic}")
        start_time = time.time()

        # Filter to top sources with content
        valid_sources = [s for s in sources if s.content and not s.error]
        valid_sources = valid_sources[:self.max_sources]

        if not valid_sources:
            logger.warning("No valid sources to synthesize")
            return SynthesizedResearch(
                topic=topic,
                summary="No sources available for synthesis.",
                confidence=0.0,
                synthesis_time_ms=0,
                raw_source_count=len(sources)
            )

        # Format sources for LLM
        formatted_sources = self._format_sources(valid_sources)

        # Call LLM for synthesis
        try:
            result = self._call_llm(topic, formatted_sources)
        except Exception as e:
            logger.error(f"LLM synthesis failed: {e}")
            # Fallback to basic synthesis
            result = self._fallback_synthesis(topic, valid_sources)

        synthesis_time = int((time.time() - start_time) * 1000)

        return SynthesizedResearch(
            topic=topic,
            summary=result.get("summary", "Synthesis failed."),
            key_insights=result.get("key_insights", []),
            sources_used=[s.url for s in valid_sources],
            confidence=result.get("confidence", 0.5),
            synthesis_time_ms=synthesis_time,
            raw_source_count=len(sources)
        )

    def _format_sources(self, sources: List[ResearchSource]) -> str:
        """Format sources for the LLM prompt."""
        parts = []
        for i, source in enumerate(sources, 1):
            content = source.content[:self.max_content_per_source]
            if len(source.content) > self.max_content_per_source:
                content += "..."

            parts.append(f"""
--- SOURCE {i}: {source.title} ---
URL: {source.url}
CONTENT:
{content}
""")
        return "\n".join(parts)

    def _call_llm(self, topic: str, sources: str) -> Dict[str, Any]:
        """Call Ollama for synthesis."""
        import requests

        prompt = self.SYNTHESIS_PROMPT.format(topic=topic, sources=sources)

        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.4,
                    "num_predict": 2000
                }
            },
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        text = result.get("response", "").strip()

        # Parse JSON from response
        try:
            if text.startswith("{"):
                return json.loads(text)
            # Try to find JSON in response
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group())
            return {"summary": text, "key_insights": [], "confidence": 0.5}
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse synthesis as JSON: {e}")
            return {"summary": text, "key_insights": [], "confidence": 0.5}

    def _fallback_synthesis(
        self,
        topic: str,
        sources: List[ResearchSource]
    ) -> Dict[str, Any]:
        """Fallback synthesis when LLM fails."""
        # Create basic summary from key points
        all_points = []
        for source in sources:
            all_points.extend(source.key_points[:2])

        if all_points:
            summary = f"Research on {topic} found the following:\n\n"
            summary += "\n".join(f"• {point}" for point in all_points[:5])
        else:
            summary = f"Found {len(sources)} sources about {topic}. "
            summary += "Key content from: " + ", ".join(s.domain for s in sources)

        return {
            "summary": summary,
            "key_insights": all_points[:3],
            "confidence": 0.4
        }


# CLI for testing
if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser(description="Test research synthesis")
    parser.add_argument("topic", nargs="?", default="event sourcing patterns", help="Topic to research and synthesize")
    parser.add_argument("--max-sources", "-n", type=int, default=3, help="Max sources")

    args = parser.parse_args()

    # First do the research
    try:
        from deep_researcher import DeepResearcher
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent))
        from deep_researcher import DeepResearcher

    print(f"\n{'='*60}")
    print(f"Research + Synthesis: {args.topic}")
    print(f"{'='*60}")

    # Step 1: Research
    print("\n[1/2] Researching...")
    researcher = DeepResearcher(max_sources=args.max_sources)
    sources = researcher.research(args.topic)

    if not sources:
        print("\nNo sources found. Cannot synthesize.")
        sys.exit(1)

    print(f"Found {len(sources)} sources")

    # Step 2: Synthesize
    print("\n[2/2] Synthesizing...")
    synthesizer = ResearchSynthesizer(max_sources=args.max_sources)
    result = synthesizer.synthesize(args.topic, sources)

    print(f"\n{'='*60}")
    print(result.format_for_display())
    print(f"\n{'='*60}")
    print(f"Confidence: {result.confidence}")
    print(f"Synthesis time: {result.synthesis_time_ms}ms")
    print(f"{'='*60}")
