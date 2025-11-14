from typing import Dict, Any
from ..utils.llm_client import call_llm


def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated] ..."


def write_results(
    question: str,
    parsed_answer: Dict[str, Any],
    run_result: Dict[str, Any],
    critic_report: Dict[str, Any],
    explanation: str,
    paper_meta: Dict[str, Any] | None = None,
) -> str:
    """
    Produce a Markdown Results section.

    Output:
        - Markdown string starting with "## Results"
    """
    pm = paper_meta or {}
    research_question = pm.get("research_question", question)
    hypotheses = pm.get("hypotheses", [])

    stdout = run_result.get("stdout", "")
    success = run_result.get("success", False)
    error = run_result.get("error")

    critic_issues = critic_report.get("issues", [])
    critic_issues_text = "\n".join(f"- {iss}" for iss in critic_issues) or "None"

    system_msg = {
        "role": "system",
        "content": (
            "You are writing the Results section of a scientific paper. "
            "You are given the research question, program stdout, a parsed numeric answer, "
            "a brief explanation of what the code did, and a critic report about consistency.\n\n"
            "Requirements:\n"
            "- Begin with a level-2 heading: '## Results'.\n"
            "- Use ONLY ASCII characters; do NOT use Unicode symbols such as π or integral signs, "
            "or smart quotes. Use LaTeX math syntax instead when needed.\n"
            "- Clearly state the main numerical results produced by the experiments, including "
            "the primary answer and any key error metrics or convergence behaviours when they "
            "can be inferred from the outputs.\n"
            "- For each major experiment or figure (for example, error vs step size, error vs "
            "number of samples, runtime comparison), describe qualitatively what the plots show "
            "and how they support or refute the hypotheses or expectations.\n"
            "- Explicitly compare numerical results to analytic or expected values when such "
            "values are known, commenting on agreement or discrepancy.\n"
            "- Summarise any notable trends: convergence rates, asymptotic behaviour, runtime "
            "scaling, or trade-offs between accuracy and cost.\n"
            "- If the critic report mentions inconsistencies or issues (for example, mismatch "
            "between printed value and explanation), briefly acknowledge them and clarify the "
            "correct interpretation of the results.\n"
            "- Do NOT describe algorithmic implementation details; those belong in Methods.\n"
            "- You may structure the narrative with level-3 subsections using '###' headings "
            "(for example, '### Primary Numerical Result', '### Convergence Behaviour', "
            "'### Runtime Comparison').\n"
            "- Target length: at least 700 words and ideally 800–900 words; do not be shorter "
            "than 600 words.\n"
            "- Aim for 3–6 paragraphs, each focused on a coherent aspect of the results.\n"
        ),
    }

    user_content = (
        f"Original user question:\n{question}\n\n"
        f"Framed research question:\n{research_question}\n\n"
        f"Hypotheses:\n{hypotheses}\n\n"
        "Program stdout:\n"
        "----- STDOUT START -----\n"
        f"{_shorten(stdout, 1000)}\n"
        "----- STDOUT END -------\n\n"
        "Parsed answer object (from stdout):\n"
        f"{parsed_answer}\n\n"
        "Success flag and error (if any):\n"
        f"success={success}, error={error}\n\n"
        "Existing short explanation of what the code did:\n"
        "----- EXPLANATION START -----\n"
        f"{_shorten(explanation, 2000)}\n"
        "----- EXPLANATION END -------\n\n"
        "Critic report summary:\n"
        f"{critic_report.get('summary', '')}\n\n"
        "Critic issues (if any):\n"
        f"{critic_issues_text}\n\n"
        "Write only the Results section in Markdown, starting with '## Results'."
    )

    user_msg = {"role": "user", "content": user_content}

    results_md = call_llm([system_msg, user_msg])
    return results_md.strip()