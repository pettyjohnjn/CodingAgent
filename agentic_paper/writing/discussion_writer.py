from typing import Dict, Any
from ..utils.llm_client import call_llm


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated] ..."


def write_discussion(
    question: str,
    methods_text: str,
    results_text: str,
    parsed_answer: Dict[str, Any],
    critic_report: Dict[str, Any],
    paper_meta: Dict[str, Any] | None = None,
) -> str:
    """
    Produce a Markdown Discussion/Conclusion section.

    Output:
        - Markdown string starting with "## Discussion and Conclusion"
    """
    pm = paper_meta or {}
    research_question = pm.get("research_question", question)
    hypotheses = pm.get("hypotheses", [])
    contributions = pm.get("contributions", [])

    critic_summary = critic_report.get("summary", "")
    critic_issues = critic_report.get("issues", [])
    critic_issues_text = "\n".join(f"- {iss}" for iss in critic_issues) or "None"

    system_msg = {
        "role": "system",
        "content": (
            "You are writing the Discussion and Conclusion section of a scientific paper. "
            "You are given the research question, Methods, Results, a parsed numeric answer, "
            "and a critic report.\n\n"
            "Requirements:\n"
            "- Begin with a level-2 heading: '## Discussion and Conclusion'.\n"
            "- Use ONLY ASCII characters; do NOT use Unicode symbols such as π or integral signs, "
            "or smart quotes. Use LaTeX math syntax instead when needed.\n"
            "- Interpret the main numerical results in the context of the research question and "
            "theoretical expectations (for example, do the observed errors match the predicted "
            "convergence rates? do deterministic methods outperform Monte Carlo in the expected way?).\n"
            "- Explicitly summarise what the experiments confirm, what they clarify, and what "
            "limitations or caveats remain.\n"
            "- Comment on the reliability and robustness of the results, including any issues "
            "raised by the critic report (for example, minor inconsistencies in rounding or "
            "limitations of the experimental setup).\n"
            "- Discuss practical implications: when the studied method or approach should be "
            "preferred in applications, and when alternative techniques might be more appropriate.\n"
            "- Clearly separate (a) what is strongly supported by the experiments from "
            "(b) speculative interpretations or hypotheses.\n"
            "- Include a concise bullet list or short paragraph that summarises the main "
            "takeaway messages of the paper.\n"
            "- Conclude with specific suggestions for future work (for example, more complex "
            "integrands, higher-dimensional problems, different error metrics, or alternative "
            "algorithms), making clear how they build on the current study.\n"
            "- Do NOT introduce new numerical results; focus on interpretation.\n"
            "- You may organise the section with level-3 subsections using '###' headings "
            "(for example, '### Interpretation of Findings', '### Limitations', "
            "'### Future Work').\n"
            "- Target length: at least 600 words and ideally 700–800 words; do not be shorter "
            "than 500 words.\n"
            "- Aim for 3–5 paragraphs of narrative text, plus any bullet list you include.\n"
        ),
    }

    user_content = (
        f"Original user question:\n{question}\n\n"
        f"Framed research question:\n{research_question}\n\n"
        f"Hypotheses:\n{hypotheses}\n\n"
        f"Planned contributions:\n{contributions}\n\n"
        "Methods section (for context):\n"
        "----- METHODS START -----\n"
        f"{_shorten(methods_text, 2000)}\n"
        "----- METHODS END -------\n\n"
        "Results section (for context):\n"
        "----- RESULTS START -----\n"
        f"{_shorten(results_text, 2000)}\n"
        "----- RESULTS END -------\n\n"
        "Parsed answer object (from stdout):\n"
        f"{parsed_answer}\n\n"
        "Critic report summary:\n"
        f"{critic_summary}\n\n"
        "Critic issues (if any):\n"
        f"{critic_issues_text}\n\n"
        "Write only the Discussion and Conclusion section in Markdown, starting with "
        "'## Discussion and Conclusion'."
    )

    user_msg = {"role": "user", "content": user_content}

    discussion_md = call_llm([system_msg, user_msg])
    return discussion_md.strip()