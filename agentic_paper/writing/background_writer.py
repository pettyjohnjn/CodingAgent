from ..utils.llm_client import call_llm


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated] ..."


def write_background(
    question: str,
    introduction_text: str,
    methods_text: str,
    results_text: str,
) -> str:
    """
    Produce a Markdown Background section.

    Output:
        - Markdown string starting with "## Background"
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are writing the Background section of a scientific paper. "
            "You are given the research question and draft Introduction, Methods, and Results sections.\n\n"
            "Requirements:\n"
            "- Begin with a level-2 heading: '## Background'.\n"
            "- Use ONLY ASCII characters; do NOT use Unicode symbols such as pi or integral signs, "
            "or smart quotes. Use LaTeX math syntax instead when needed.\n"
            "- Provide the theoretical and conceptual foundations needed to understand the "
            "problem and methods in this paper.\n"
            "- Explain the core mathematical or algorithmic concepts at a textbook level "
            "(for example, classical quadrature rules, convergence orders, error notions, "
            "Monte Carlo basics), but do NOT repeat all low-level algorithmic details that "
            "will appear in Methods.\n"
            "- Connect the background concepts directly to the specific problem formulation "
            "and research question of this work.\n"
            "- You may introduce 1–2 short level-3 subsections using '###' headings "
            "(for example, '### Numerical Integration Basics', '### Error and Convergence').\n"
            "- Do NOT describe this paper's specific experiments or numerical results; "
            "those belong in Methods and Results.\n"
            "- Do NOT survey specific prior papers in detail; that belongs in Related Work.\n"
            "- Target length: at least 500 words and ideally 600–700 words; do not be shorter "
            "than 450 words.\n"
            "- Aim for 3–5 paragraphs with clear, pedagogical explanations.\n"
        ),
    }

    user_content = (
        f"Research question:\n{question}\n\n"
        "Draft Introduction section (for context):\n"
        "----- INTRO START -----\n"
        f"{_shorten(introduction_text, 2000)}\n"
        "----- INTRO END -------\n\n"
        "Draft Methods section (for context):\n"
        "----- METHODS START ----\n"
        f"{_shorten(methods_text, 2000)}\n"
        "----- METHODS END ------\n\n"
        "Draft Results section (for context):\n"
        "----- RESULTS START ----\n"
        f"{_shorten(results_text, 2000)}\n"
        "----- RESULTS END ------\n\n"
        "Write only the Background section in Markdown, starting with '## Background'."
    )

    user_msg = {"role": "user", "content": user_content}

    background_md = call_llm([system_msg, user_msg])
    return background_md.strip()