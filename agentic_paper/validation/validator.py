import re
from typing import Tuple, List, Optional
from ..config import AgentConfig


def validate_code(
    code: str,
    config: AgentConfig,
    require_answer_print: Optional[bool] = None,
) -> Tuple[bool, List[str]]:
    """Validate generated code against simple safety and contract rules.

    Returns (is_valid, list_of_issues).

    `require_answer_print` overrides the config flag if provided.
    """
    issues: List[str] = []

    # Check forbidden modules
    for mod in config.forbidden_modules:
        if re.search(rf"\bimport\s+{re.escape(mod)}\b", code) or re.search(
            rf"\bfrom\s+{re.escape(mod)}\b", code
        ):
            issues.append(f"Use of forbidden module '{mod}' is not allowed.")

    # Check forbidden calls
    for call in config.forbidden_calls:
        if call in code:
            issues.append(f"Use of forbidden call '{call}' is not allowed.")

    # NEW: reject markdown code fences so they never reach exec()
    if "```" in code:
        issues.append(
            "Response contains markdown code fences (```); respond with plain Python code only."
        )

    # Check that the code prints an 'Answer:' line when required
    need_answer = config.require_answer_print if require_answer_print is None else require_answer_print
    if need_answer:
        if not re.search(r"print\s*\(\s*['\"]Answer:\s*", code):
            issues.append(
                "Code must print the final answer with a line like: print('Answer:', value)"
            )

    return (len(issues) == 0, issues)