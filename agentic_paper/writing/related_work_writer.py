from ..utils.llm_client import call_llm


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated] ..."


def write_related_work(
    question: str,
    introduction_text: str,
    background_text: str,
    methods_text: str,
    results_text: str,
) -> str:
    """
    Produce a Markdown Related Work section.

    Output:
        - Markdown string starting with "## Related Work"
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are writing the Related Work section of a scientific paper. "
            "You are given the research question and draft Introduction, Background, "
            "Methods, and Results sections.\n\n"
            "Requirements:\n"
            "- Begin with a level-2 heading: '## Related Work'.\n"
            "- Use ONLY ASCII characters; do NOT use Unicode symbols such as pi or integral signs, "
            "or smart quotes. Use LaTeX math syntax instead when needed.\n"
            "- Provide a structured survey of prior work that is relevant to the problem and "
            "methods in this paper.\n"
            "- Group prior work into 2–4 thematic threads (for example, 'classical numerical "
            "analysis of quadrature', 'Monte Carlo and quasi-Monte Carlo methods', "
            "'modern scientific computing libraries and frameworks').\n"
            "- For each thread, describe what is known, what is commonly done, and how this "
            "paper is similar to or different from that body of work.\n"
            "- Refer generically to prior work (for example, 'classical textbooks in numerical "
            "analysis', 'standard references on Monte Carlo methods') rather than fabricating "
            "very specific paper titles or citation details.\n"
            "- Explicitly explain how this paper positions itself relative to these strands of "
            "work: what gap it addresses, what it confirms, what it extends, or what it "
            "simplifies.\n"
            "- You may introduce level-3 subsections using '###' headings to organise the "
            "discussion.\n"
            "- Do NOT restate this paper's own contributions in detail; they belong in the "
            "Introduction and Discussion.\n"
            "- Do NOT present new numerical results here; those belong in Results.\n"
            "- Target length: at least 600 words and ideally 700–900 words; do not be shorter "
            "than 550 words.\n"
            "- Aim for 3–6 paragraphs, each focused on a coherent subset of the literature.\n"
        ),
    }

    user_content = (
        f"Research question:\n{question}\n\n"
        "Draft Introduction section (for context):\n"
        "----- INTRO START -----\n"
        f"{_shorten(introduction_text, 2000)}\n"
        "----- INTRO END -------\n\n"
        "Draft Background section (for context):\n"
        "----- BACKGROUND START -----\n"
        f"{_shorten(background_text, 2000)}\n"
        "----- BACKGROUND END -------\n\n"
        "Draft Methods section (for context):\n"
        "----- METHODS START ----\n"
        f"{_shorten(methods_text, 2000)}\n"
        "----- METHODS END ------\n\n"
        "Draft Results section (for context):\n"
        "----- RESULTS START ----\n"
        f"{_shorten(results_text, 2000)}\n"
        "----- RESULTS END ------\n\n"
        "Write only the Related Work section in Markdown, starting with '## Related Work'."
    )

    user_msg = {"role": "user", "content": user_content}

    related_md = call_llm([system_msg, user_msg])
    return related_md.strip()