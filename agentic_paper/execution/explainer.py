
from typing import Dict, Any
from ..utils.llm_client import call_llm

def generate_explanation(
    question: str,
    combined_code: str,
    run_result: Dict[str, Any],
) -> str:
    """Ask the LLM to explain the code and its output, and clearly state the answer."""
    system_msg = {
        "role": "system",
        "content": (
            "You are a concise scientific and programming explainer. "
            "Given a question, some Python code that was run, and its output, "
            "explain in 1–3 short paragraphs what the code did and what the answer is.\n"
            "If there was an error, explain what went wrong instead of a final answer.\n"
            "End by clearly stating the final answer (or that no answer was obtained)."
        ),
    }

    stdout = run_result.get("stdout", "")
    error = run_result.get("error")
    success = run_result.get("success", False)

    # Only show a snippet of stdout if it's very long
    max_len = 2000
    if len(stdout) > max_len:
        stdout_snippet = stdout[:max_len] + "\n... [truncated] ..."
    else:
        stdout_snippet = stdout

    # Also possibly truncate code if enormous
    code_snippet = combined_code
    if len(code_snippet) > 4000:
        code_snippet = code_snippet[:4000] + "\n... [code truncated] ..."

    user_content = (
        f"Question:\n{question}\n\n"
        "Here is the Python code that was generated and executed "
        "(possibly from multiple files concatenated):\n\n"
        "----- CODE START -----\n"
        f"{code_snippet}\n"
        "----- CODE END -------\n\n"
        "Here is the captured stdout:\n"
        "----- STDOUT START ---\n"
        f"{stdout_snippet}\n"
        "----- STDOUT END -----\n\n"
        f"Execution success flag: {success}\n"
        f"Error (if any): {error}\n\n"
        "Explain what the code did and what the answer to the question is. "
        "If an error occurred, explain what went wrong and what should be fixed."
    )

    user_msg = {
        "role": "user",
        "content": user_content,
    }

    explanation = call_llm([system_msg, user_msg])
    return explanation.strip()
