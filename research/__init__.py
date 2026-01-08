"""
Senter Autonomous Research Module

Provides intelligent topic detection, web research, and synthesis.
"""

# Import available modules (others added as implemented)
from .topic_extractor import TopicExtractor, ExtractedTopic

__all__ = [
    "TopicExtractor",
    "ExtractedTopic",
]

# Conditional imports for modules being developed
try:
    from .deep_researcher import DeepResearcher, ResearchSource
    __all__.extend(["DeepResearcher", "ResearchSource"])
except ImportError:
    pass

try:
    from .synthesizer import ResearchSynthesizer, SynthesizedResearch
    __all__.extend(["ResearchSynthesizer", "SynthesizedResearch"])
except ImportError:
    pass

try:
    from .research_store import ResearchStore
    __all__.append("ResearchStore")
except ImportError:
    pass

try:
    from .pipeline import ResearchPipeline
    __all__.append("ResearchPipeline")
except ImportError:
    pass
