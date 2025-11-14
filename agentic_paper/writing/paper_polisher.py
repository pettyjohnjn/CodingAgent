from typing import Tuple

from .latex_style import LATEX_STYLE_HINTS
from ..codegen.codegen import call_llm  # or your client wrapper


def polish_paper_tex(draft_tex: str, model: str) -> str:
    """
    One-pass LaTeX polishing using the style hints.
    """
    system_prompt = (
        "You are an expert LaTeX copy-editor. "
        "You receive a single LaTeX document and must return a polished version. "
        "Follow the style hints exactly."
    )
    user_prompt = (
        LATEX_STYLE_HINTS
        + "\n\nHere is the LaTeX document to polish. "
        "Return ONLY LaTeX code, with the same overall structure:\n\n"
        + draft_tex
    )
    return call_llm(model=model, system_prompt=system_prompt, user_prompt=user_prompt)