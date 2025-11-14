import re
from typing import Dict, Any


def parse_answer_from_stdout(stdout: str) -> Dict[str, Any]:
    """
    Parse the numeric answer from the agent's stdout.

    We look for the last line containing "Answer:" and then try to extract
    a floating-point number from that line.

    Returns a dict:
        {
            "raw_line": str | None,
            "value": float | None,
            "parse_success": bool,
            "error": str | None,
        }
    """
    lines = stdout.splitlines()
    answer_line = None

    for line in reversed(lines):
        if "Answer:" in line:
            answer_line = line.strip()
            break

    if answer_line is None:
        return {
            "raw_line": None,
            "value": None,
            "parse_success": False,
            "error": "No line containing 'Answer:' found in stdout.",
        }

    # Try to extract a number after "Answer:"
    # Matches int/float/float with exponent, e.g. 1, -3.14, 1.23e-4
    number_pattern = re.compile(
        r"Answer:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    )
    m = number_pattern.search(answer_line)
    if not m:
        return {
            "raw_line": answer_line,
            "value": None,
            "parse_success": False,
            "error": "Could not parse a numeric value after 'Answer:'.",
        }

    num_str = m.group(1)
    try:
        value = float(num_str)
    except Exception as e:
        return {
            "raw_line": answer_line,
            "value": None,
            "parse_success": False,
            "error": f"Failed to convert '{num_str}' to float: {e!r}",
        }

    return {
        "raw_line": answer_line,
        "value": value,
        "parse_success": True,
        "error": None,
    }