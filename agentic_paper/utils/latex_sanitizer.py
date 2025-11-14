"""
Utilities for cleaning and converting text into LaTeX-safe form.

Responsibilities:
- Normalize Unicode to a plain ASCII representation suitable for pdflatex.
- Replace common math-related Unicode symbols with LaTeX equivalents.
- Provide a minimal Markdown-to-LaTeX converter for section bodies.
"""

from __future__ import annotations

import re
import unicodedata


# Map specific Unicode characters to LaTeX-friendly ASCII strings.
# This runs AFTER Unicode normalization so most fancy letters are already plain.
UNICODE_TO_LATEX = {
    # spaces / quotes / dashes
    "\u00a0": " ",   # non-breaking space
    "\u202f": " ",   # narrow no-break space
    "\u2007": " ",   # figure space
    "\u2009": " ",   # thin space
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‚": "'",
    "‛": "'",
    "–": "--",       # en dash
    "—": "---",      # em dash
    "−": "-",        # minus sign
    "…": "...",

    # math-ish symbols
    "×": r"$\times$",
    "·": r"$\cdot$",
    "•": r"$\cdot$",
    "≤": r"$\le$",
    "≥": r"$\ge$",
    "≃": r"$\simeq$",
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "±": r"$\pm$",
    "∓": r"$\mp$",
    "∞": r"$\infty$",
    "∫": r"$\int$",
    "∑": r"$\sum$",
    "√": r"$\sqrt{}$",
    "π": r"$\pi$",
    "Π": r"$\Pi$",

    # arrows
    "→": r"$\to$",
    "←": r"$\leftarrow$",
    "↔": r"$\leftrightarrow$",
    "⇒": r"$\Rightarrow$",
    "⇐": r"$\Leftarrow$",
    "⇔": r"$\Leftrightarrow$",

    # a few superscripts/subscripts that sometimes survive normalization
    "²": r"$^2$",
    "³": r"$^3$",
    "⁴": r"$^4$",
    "⁵": r"$^5$",
    "⁶": r"$^6$",
    "⁷": r"$^7$",
    "⁸": r"$^8$",
    "⁹": r"$^9$",
    "⁰": r"$^0$",
    "₀": r"$_0$",
    "₁": r"$_1$",
    "₂": r"$_2$",
    "₃": r"$_3$",
    "₄": r"$_4$",
    "₅": r"$_5$",
    "₆": r"$_6$",
    "₇": r"$_7$",
    "₈": r"$_8$",
    "₉": r"$_9$",
}


def _apply_unicode_map(text: str) -> str:
    """Apply the UNICODE_TO_LATEX substitutions."""
    for src, tgt in UNICODE_TO_LATEX.items():
        text = text.replace(src, tgt)
    return text


def sanitize_for_latex(text: str) -> str:
    """
    Normalize and sanitize text for LaTeX:

    - Unicode NFKC normalization (turns math alphanumerics into plain ASCII where possible).
    - Replace a set of known Unicode characters with LaTeX or ASCII equivalents.
    - Drop any remaining non-ASCII characters (outside newline/tab).
    """
    # Normalize fancy math letters, superscripts etc. to a canonical form
    normalized = unicodedata.normalize("NFKC", text)

    # Apply explicit mapping of troublesome characters
    normalized = _apply_unicode_map(normalized)

    # Final pass: strip any remaining non-ASCII (except newline/tab)
    cleaned_chars = []
    for ch in normalized:
        if ord(ch) < 128 or ch in "\n\t":
            cleaned_chars.append(ch)
        # else: drop it silently
    cleaned = "".join(cleaned_chars)
    return cleaned


def _escape_underscores_in_command(text: str, command: str) -> str:
    """
    Escape underscores inside the argument of a simple LaTeX command
    like \texttt{...}, \textbf{...}, or \emph{...}.

    Example:
        \texttt{midpoint_rule} -> \texttt{midpoint\_rule}
    """
    pattern = re.compile(rf"(\\{command}\{{)([^{{}}]*)(\}})")

    def repl(match: re.Match) -> str:
        prefix, body, suffix = match.groups()
        body_escaped = body.replace("_", r"\_")
        return prefix + body_escaped + suffix

    return pattern.sub(repl, text)


def markdown_to_latex_body(md: str) -> str:
    """
    Minimal Markdown-to-LaTeX conversion for section bodies.

    Steps:
    - Sanitize Unicode to ASCII/LaTeX-safe form.
    - Convert:
        `code`   -> \texttt{code}
        **bold** -> \textbf{bold}
        *italic* -> \emph{italic}
    - Escape underscores inside \texttt{...}, \textbf{...}, and \emph{...}
      so variable-like names (midpoint_rule) do not break LaTeX.
    - Leave explicit LaTeX math (\(...\), \[...\]) untouched (they are ASCII).
    """
    text = sanitize_for_latex(md)

    # Inline code: `code` -> \texttt{code}
    text = re.sub(
        r"`([^`]+)`",
        r"\\texttt{\1}",
        text,
    )

    # Bold: **text** -> \textbf{text}
    text = re.sub(
        r"\*\*([^*]+)\*\*",
        r"\\textbf{\1}",
        text,
    )

    # Italic: *text* -> \emph{text} (avoid **bold** clashes)
    text = re.sub(
        r"(?<!\*)\*([^*]+)\*(?!\*)",
        r"\\emph{\1}",
        text,
    )

    # Escape underscores inside simple text commands
    for cmd in ("texttt", "textbf", "emph"):
        text = _escape_underscores_in_command(text, cmd)

    return text