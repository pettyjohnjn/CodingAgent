import re

UNICODE_TO_LATEX = {
    "–": "--",
    "—": "---",
    "−": "-",   # Unicode minus
    "“": "``",
    "”": "''",
    "‘": "'",
    "’": "'",
    "…": "\\ldots{}",
    "\u00a0": " ",  # non-breaking space
}

def sanitize_latex(text: str) -> str:
    """Normalize Unicode punctuation and a few common LaTeX gotchas."""
    for bad, good in UNICODE_TO_LATEX.items():
        text = text.replace(bad, good)

    # Remove any stray Markdown fences if they ever appear
    text = text.replace("```latex", "").replace("```tex", "").replace("```", "")

    # Optionally deduplicate figure labels by appending suffixes.
    # This is conservative: only changes exact duplicate labels.
    seen = {}

    def _fix_label(match: re.Match) -> str:
        full = match.group(0)
        name = match.group(1)
        count = seen.get(name, 0)
        seen[name] = count + 1
        if count == 0:
            return full
        return f"\\label{{{name}-{count}}}"

    text = re.sub(r"\\label\{([^}]+)\}", _fix_label, text)

    return text