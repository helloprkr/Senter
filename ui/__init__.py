"""
Senter UI Module

macOS menubar app with research panel.
"""

# Conditional imports
try:
    from .menubar_app import SenterMenubar
    __all__ = ["SenterMenubar"]
except ImportError:
    pass

try:
    from .research_panel import ResearchPanel
    __all__.append("ResearchPanel")
except ImportError:
    pass
