import re
from typing import Dict, Any, List


def _parse_numeric_from_text(text: str) -> Dict[str, Any]:
    """
    Heuristically extract a numeric value from a block of text.

    We look for all numeric tokens and take the LAST one, assuming that
    the explanation typically ends with the final answer.

    Returns:
        {
            "value": float | None,
            "token": str | None,
            "parse_success": bool,
            "error": str | None,
        }
    """
    # Match ints/floats/exp notation
    num_pattern = re.compile(
        r"([-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    matches = num_pattern.findall(text)
    if not matches:
        return {
            "value": None,
            "token": None,
            "parse_success": False,
            "error": "No numeric token found in text.",
        }

    token = matches[-1]
    try:
        value = float(token)
    except Exception as e:
        return {
            "value": None,
            "token": token,
            "parse_success": False,
            "error": f"Failed to convert '{token}' to float: {e!r}",
        }

    return {
        "value": value,
        "token": token,
        "parse_success": True,
        "error": None,
    }


def simple_critic(
    question: str,
    parsed_answer: Dict[str, Any],
    explanation: str,
    run_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Simple critic that checks consistency between stdout's parsed answer
    and the numeric value inferred from the explanation text.

    Returns a dict with:
        {
            "summary": str,
            "issues": List[str],
            "parsed_answer": parsed_answer,
            "explanation_numeric": {...},
            "consistency_ok": bool,
        }
    """
    issues: List[str] = []

    # If we couldn't parse the answer from stdout, that's already an issue.
    if not parsed_answer.get("parse_success", False):
        issues.append(
            f"Failed to parse numeric answer from stdout: {parsed_answer.get('error')}"
        )

    # Parse numeric from explanation text
    explanation_numeric = _parse_numeric_from_text(explanation)
    if not explanation_numeric.get("parse_success", False):
        issues.append(
            f"Failed to parse numeric value from explanation: "
            f"{explanation_numeric.get('error')}"
        )

    consistency_ok = False
    if parsed_answer.get("parse_success") and explanation_numeric.get("parse_success"):
        v_stdout = parsed_answer["value"]
        v_expl = explanation_numeric["value"]

        # Relative difference check
        denom = max(abs(v_stdout), 1.0)
        rel_diff = abs(v_stdout - v_expl) / denom

        if rel_diff <= 1e-6:
            consistency_ok = True
        else:
            issues.append(
                "Numeric value in explanation differs significantly from stdout "
                f"(stdout={v_stdout}, explanation={v_expl}, rel_diff={rel_diff})."
            )

    # Also treat absence of an 'Answer:' line as an issue
    stdout = run_result.get("stdout", "")
    if "Answer:" not in stdout:
        issues.append("Stdout does not contain an 'Answer:' line.")

    if not issues:
        summary = (
            "Explanation appears consistent with the parsed numeric answer from stdout."
        )
    else:
        summary = "Critic found potential issues with the result or explanation."

    return {
        "summary": summary,
        "issues": issues,
        "parsed_answer": parsed_answer,
        "explanation_numeric": explanation_numeric,
        "consistency_ok": consistency_ok,
    }