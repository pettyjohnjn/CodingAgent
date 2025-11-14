from typing import Dict, Any

from ..utils.llm_client import call_llm


def write_introduction(
    question: str,
    methods_text: str,
    results_text: str,
    paper_meta: Dict[str, Any] | None = None,
) -> str:
    """
    Produce a Markdown Introduction section.

    Output:
        - Markdown string starting with "## Introduction"
    """
    pm = paper_meta or {}
    research_question = pm.get("research_question", question)
    hypotheses = pm.get("hypotheses", [])
    contributions = pm.get("contributions", [])
    domain_context = pm.get("domain_context", "")
    novelty_frame = pm.get("novelty_frame", [])

    system_msg = {
        "role": "system",
        "content": (
            "You are writing the Introduction section of a scientific paper. "
            "You are given the research question and draft Methods and Results sections.\n\n"
            "Requirements:\n"
            "- Begin with a level-2 heading: '## Introduction'.\n"
            "- Use ONLY ASCII characters; do NOT use Unicode symbols such as π or integral signs, "
            "or smart quotes. Use LaTeX math syntax instead when needed (for example, "
            "I = \\int_0^\\pi \\sin x\\,dx).\n"
            "- Clearly introduce the broader problem area and why it matters (1–2 paragraphs).\n"
            "- Narrow down to the specific problem studied in this paper and state the main "
            "research question explicitly.\n"
            "- Include a short high-level description of the technical approach, but do NOT give "
            "implementation details (those belong in Methods).\n"
            "- Explicitly articulate what is novel or distinctive about this work compared to "
            "standard textbook treatments.\n"
            "- Include a clearly marked contributions list as a bullet list introduced by a "
            "sentence such as 'This work makes the following contributions:'.\n"
            "- You may introduce 1–2 short level-3 subsections using '###' headings if helpful "
            "(for example, '### Problem Setting', '### Contributions').\n"
            "- Do NOT include specific numerical results or detailed error values; those belong "
            "in the Results section.\n"
            "- Target length: at least 700 words and ideally 800–900 words; do not be shorter "
            "than 600 words.\n"
            "- Aim for 4–8 paragraphs (excluding the bullet list), with smooth transitions "
            "between ideas.\n"
        ),
    }

    user_content = (
        f"Original user question:\n{question}\n\n"
        f"Framed research question:\n{research_question}\n\n"
        f"Domain context:\n{domain_context}\n\n"
        f"Hypotheses (list):\n{hypotheses}\n\n"
        f"Novelty framing phrases:\n{novelty_frame}\n\n"
        f"Planned contributions (list):\n{contributions}\n\n"
        "Draft Methods section (for context, do not duplicate):\n"
        "----- METHODS START -----\n"
        f"{methods_text}\n"
        "----- METHODS END -------\n\n"
        "Draft Results section (for context, do not duplicate):\n"
        "----- RESULTS START -----\n"
        f"{results_text}\n"
        "----- RESULTS END -------\n\n"
        "Write only the Introduction section in Markdown, starting with '## Introduction'."
    )

    user_msg = {"role": "user", "content": user_content}

    intro_md = call_llm([system_msg, user_msg])
    return intro_md.strip()