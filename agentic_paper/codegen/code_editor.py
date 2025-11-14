
from typing import Dict, Any, List
from ..utils.llm_client import call_llm

def revise_code(
    question: str,
    previous_code: str,
    run_result: Dict[str, Any],
    validation_issues: List[str],
) -> str:
    system_msg = {
        "role": "system",
        "content": (
            "You are a code-fixing assistant. You receive a Python script, "
            "its runtime result (including any error), and a list of validation issues. "
            "Your task is to produce a corrected FULL Python script that:\n"
            "- Is self-contained and executable in the shared namespace with other files.\n"
            "- Avoids forbidden modules/calls mentioned in the validation issues.\n"
            "- Prints the final answer with a line like: print('Answer:', value).\n"
            "Respond with ONLY valid Python code, no markdown or explanations."
        ),
    }

    stdout = run_result.get("stdout", "")
    error = run_result.get("error")

    # shorten stdout if huge
    if len(stdout) > 2000:
        stdout = stdout[:2000] + "\n... [truncated] ..."

    issues_formatted = "\n".join(f"- {iss}" for iss in validation_issues) or "None"

    user_content = (
        f"Question:\n{question}\n\n"
        "Here is the previous Python script (for the entrypoint file):\n"
        "----- CODE START -----\n"
        f"{previous_code}\n"
        "----- CODE END -------\n\n"
        "Here is the runtime result of that script (stdout and error):\n"
        f"stdout:\n{stdout}\n\n"
        f"error: {error}\n\n"
        "Here are the validation issues you must fix:\n"
        f"{issues_formatted}\n\n"
        "Produce a corrected full Python script that addresses these problems. "
        "Remember: output ONLY Python code, no backticks or explanations."
    )

    user_msg = {"role": "user", "content": user_content}

    new_code = call_llm([system_msg, user_msg])
    return new_code.strip()
