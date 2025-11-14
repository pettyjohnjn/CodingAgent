from typing import Dict, Any, List
from ..utils.llm_client import call_llm

def _shorten(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "\n... [truncated] ..."


def write_methods(
    question: str,
    project_plan: Dict[str, Any],
    code_by_file: Dict[str, str],
    parsed_answer: Dict[str, Any],
    run_result: Dict[str, Any],
    paper_meta: Dict[str, Any] | None = None,
) -> str:
    """
    Produce a Markdown Methods section describing:

    - What problem was framed as a research question.
    - What numerical methods / algorithms were used.
    - How helper functions and entrypoint interact.
    - Key parameters (n, step size, etc.) if they appear in code.
    - How the experiments operationalize the hypotheses when appropriate.

    Returns a Markdown string (no surrounding backticks).
    """
    pm = paper_meta or {}
    research_question = pm.get("research_question", question)
    hypotheses = pm.get("hypotheses", [])
    contributions = pm.get("contributions", [])
    domain_context = pm.get("domain_context", "")

    # Collect a compact description of project files and roles
    files_desc_lines: List[str] = []
    for f in project_plan.get("files", []):
        files_desc_lines.append(
            f"- {f['name']} (role={f.get('role')}, desc={f.get('description', '')})"
        )
    files_desc = "\n".join(files_desc_lines)

    # Include helper and main code, truncated if necessary
    helpers_code = code_by_file.get("helpers.py", "")
    main_code = code_by_file.get("main.py", "")

    helpers_code_snippet = _shorten(helpers_code, 3000)
    main_code_snippet = _shorten(main_code, 3000)

    stdout = run_result.get("stdout", "")

    system_msg = {
        "role": "system",
        "content": (
            "You are writing the Methods section of a scientific paper. "
            "You are given the research question, the structure and code of a small multi-file "
            "Python project (helpers plus main), and program output.\n\n"
            "Requirements:\n"
            "- Begin with a level-2 heading: '## Methods'.\n"
            "- Use ONLY ASCII characters; do NOT use Unicode symbols such as π or integral signs, "
            "or smart quotes. Use LaTeX math syntax instead when needed.\n"
            "- Describe the numerical methods and algorithms used in enough detail that a reader "
            "could reimplement them (for example, left Riemann sum, trapezoidal rule, "
            "Simpson's rule, Monte Carlo integration), but you may omit low-level Python syntax.\n"
            "- Explain how the helper functions and the main script interact conceptually "
            "(for example, helpers implement integration routines and error computations, "
            "while the main script orchestrates experiments, loops over parameters, and "
            "produces plots).\n"
            "- Describe the experimental setup: parameter ranges, grids, step sizes, sample "
            "sizes, number of trials, and what metrics are recorded (for example, approximate "
            "values, absolute errors, runtimes).\n"
            "- Clearly state any assumptions, simplifications, or design choices in the "
            "experimental protocol (for example, fixed random seed, fixed number of "
            "Monte Carlo samples, chosen step sizes).\n"
            "- You may use level-3 subsections with '###' headings to organise the section "
            "(for example, '### Numerical Methods', '### Experimental Setup', "
            "'### Error Metrics and Figures').\n"
            "- Do NOT restate the main numerical results or detailed error values; those "
            "belong in the Results section.\n"
            "- Target length: at least 700 words and ideally 800–900 words; do not be shorter "
            "than 600 words.\n"
            "- Aim for 4–7 paragraphs, with clear, technical prose.\n"
        ),
    }

    user_content = (
        f"Original user question:\n{question}\n\n"
        f"Framed research question:\n{research_question}\n\n"
        f"Domain context:\n{domain_context}\n\n"
        f"Hypotheses:\n{hypotheses}\n\n"
        f"Planned contributions:\n{contributions}\n\n"
        "Project structure:\n"
        f"{files_desc}\n\n"
        "Here is the source code for helpers.py:\n"
        "----- helpers.py START -----\n"
        f"{helpers_code_snippet}\n"
        "----- helpers.py END -------\n\n"
        "Here is the source code for main.py:\n"
        "----- main.py START --------\n"
        f"{main_code_snippet}\n"
        "----- main.py END ----------\n\n"
        "Captured stdout (for context, not for Results):\n"
        f"{_shorten(stdout, 500)}\n\n"
        "Write only the Methods section in Markdown, starting with '## Methods'."
    )

    user_msg = {"role": "user", "content": user_content}

    methods_md = call_llm([system_msg, user_msg])
    return methods_md.strip()