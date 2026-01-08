#!/usr/bin/env python3
"""
UI Styles for Senter

Glassmorphism-inspired styling for modern macOS look.
"""

# Color palette
COLORS = {
    "background": "rgba(30, 30, 40, 0.85)",
    "surface": "rgba(45, 45, 55, 0.9)",
    "surface_hover": "rgba(55, 55, 65, 0.95)",
    "primary": "#7C3AED",  # Purple
    "primary_light": "#A78BFA",
    "secondary": "#10B981",  # Green
    "text": "#F3F4F6",
    "text_secondary": "#9CA3AF",
    "text_muted": "#6B7280",
    "border": "rgba(255, 255, 255, 0.1)",
    "shadow": "rgba(0, 0, 0, 0.3)",
    "success": "#10B981",
    "warning": "#F59E0B",
    "error": "#EF4444",
}

# Main window style
WINDOW_STYLE = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}
"""

# Panel/Card style
PANEL_STYLE = f"""
QFrame#researchCard {{
    background-color: {COLORS['surface']};
    border: 1px solid {COLORS['border']};
    border-radius: 12px;
    padding: 16px;
}}

QFrame#researchCard:hover {{
    background-color: {COLORS['surface_hover']};
    border: 1px solid {COLORS['primary']};
}}
"""

# Text styles
TEXT_STYLES = f"""
QLabel#title {{
    color: {COLORS['text']};
    font-size: 18px;
    font-weight: bold;
}}

QLabel#subtitle {{
    color: {COLORS['text_secondary']};
    font-size: 14px;
}}

QLabel#body {{
    color: {COLORS['text']};
    font-size: 13px;
    line-height: 1.5;
}}

QLabel#muted {{
    color: {COLORS['text_muted']};
    font-size: 12px;
}}

QLabel#insight {{
    color: {COLORS['text']};
    font-size: 13px;
    padding-left: 16px;
}}
"""

# Button styles
BUTTON_STYLES = f"""
QPushButton {{
    background-color: {COLORS['primary']};
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 13px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS['primary_light']};
}}

QPushButton:pressed {{
    background-color: {COLORS['primary']};
}}

QPushButton#secondary {{
    background-color: transparent;
    border: 1px solid {COLORS['border']};
    color: {COLORS['text']};
}}

QPushButton#secondary:hover {{
    background-color: {COLORS['surface']};
}}
"""

# Scroll area
SCROLL_STYLE = f"""
QScrollArea {{
    background: transparent;
    border: none;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: {COLORS['text_muted']};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_secondary']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
"""

# Rating stars
STAR_STYLE = f"""
QPushButton#star {{
    background: transparent;
    border: none;
    font-size: 20px;
    padding: 4px;
    color: {COLORS['text_muted']};
}}

QPushButton#star:hover {{
    color: {COLORS['warning']};
}}

QPushButton#starFilled {{
    background: transparent;
    border: none;
    font-size: 20px;
    padding: 4px;
    color: {COLORS['warning']};
}}
"""

# Confidence indicator
CONFIDENCE_STYLES = f"""
QProgressBar {{
    background-color: {COLORS['surface']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['secondary']};
    border-radius: 4px;
}}
"""

# Combined stylesheet
FULL_STYLESHEET = "\n".join([
    WINDOW_STYLE,
    PANEL_STYLE,
    TEXT_STYLES,
    BUTTON_STYLES,
    SCROLL_STYLE,
    STAR_STYLE,
    CONFIDENCE_STYLES,
])


def get_confidence_color(confidence: float) -> str:
    """Get color based on confidence level."""
    if confidence >= 0.8:
        return COLORS["success"]
    elif confidence >= 0.6:
        return COLORS["warning"]
    else:
        return COLORS["error"]
