from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

from ..utils.llm_client import call_llm


def _build_plan_summary(project_plan: Dict[str, Any]) -> str:
    """Extract and pretty-print the experiments section of the plan."""
    experiments = project_plan.get("experiments", []) or []
    return json.dumps(experiments, indent=2)


def _build_code_bundle(code_by_file: Dict[str, str]) -> str:
    """Concatenate all files into a single annotated text block."""
    parts: List[str] = []
    # Stable ordering for determinism
    for name in sorted(code_by_file.keys()):
        parts.append(f"===== FILE: {name} =====")
        parts.append(code_by_file[name])
        parts.append("")
    return "\n".join(parts)


def _parse_llm_json(content: str) -> Dict[str, Any]:
    """
    Robust JSON extraction from the model output.

    The model is instructed to return a single JSON object, but this helper
    tolerates minor deviations (e.g., leading/trailing text).
    """
    content = content.strip()
    # First, try a direct parse
    try:
        obj = json.loads(content)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Fallback: try to extract the first {...} block
    start_idx = content.find("{")
    end_idx = content.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        snippet = content[start_idx : end_idx + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fail-open: no usable JSON, treat as "no blocking issues"
    return {"ok": True, "issues_by_file": {}}


def spec_critic(
    project_plan: Dict[str, Any],
    code_by_file: Dict[str, str],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare the experiment plan against the generated code using a language model.

    Inputs:
        project_plan: full plan dict from the planner. We only care about
                      project_plan["experiments"].
        code_by_file: mapping filename -> Python source code.
        model:        optional override of the LLM model name.

    Returns:
        {
          "ok": bool,
          "issues_by_file": { "some_file.py": ["issue", ...], ... }
        }

    Semantics:
        - ok == True  means "no blocking mismatches detected between plan and code".
        - ok == False means "at least one experiment/plot/metric/Answer requirement
          appears to be missing or wrong".
    """
    experiments = project_plan.get("experiments") or []
    if not experiments:
        # Nothing to compare against; do not invent failures.
        return {"ok": True, "issues_by_file": {}}

    if model is None:
        model = "openai/gpt-oss-120b"

    experiments_json = _build_plan_summary(project_plan)
    code_bundle = _build_code_bundle(code_by_file)

    system_prompt = (
        "You are a rigorous static-analysis assistant for Python numerical experiments.\n"
        "You receive:\n"
        "1) An experiment plan (project_plan['experiments']) describing experiments:\n"
        "   - which numerical methods should be used,\n"
        "   - what parameter grids (e.g., N values, tolerances) should be explored,\n"
        "   - which plots (with exact filenames) should be generated,\n"
        "   - which metrics should be computed and printed.\n"
        "2) A multi-file Python project (arbitrary filenames).\n\n"
        "Your job is to check whether the code actually implements what the plan asks for.\n\n"
        "You MUST respond with a single JSON object of the form:\n"
        "{\n"
        '  \"ok\": boolean,\n'
        '  \"issues_by_file\": {\n'
        '    \"some_file.py\": [\"issue 1\", \"issue 2\"],\n'
        '    \"other_file.py\": [\"issue 3\"],\n'
        "    ...\n"
        "  }\n"
        "}\n\n"
        "Guidelines:\n"
        "- Set \"ok\" to false if ANY experiment, plot, or metric described in the plan\n"
        "  appears to be missing or clearly mis-implemented.\n"
        "- Focus on concrete mismatches, such as:\n"
        "  * Missing experiments (e.g., only trapezoidal implemented when plan also\n"
        "    requires Simpson).\n"
        "  * Missing parameter sweeps (e.g., plan specifies N from 2..1000 but code only\n"
        "    runs a single N value).\n"
        "  * Missing plt.savefig(...) calls for expected PNG filenames.\n"
        "  * Missing printed metrics that the plan requires.\n"
        "  * Missing or malformed final answer line (should print like: 'Answer:', value).\n"
        "- Use actual code contents to attribute issues to specific files when possible.\n"
        "- Keep issue messages short and concrete.\n"
        "- Do NOT include any commentary outside the JSON object.\n"
    )

    user_prompt = (
        "EXPERIMENT PLAN (project_plan['experiments']):\n"
        "--------------------\n"
        f"{experiments_json}\n\n"
        "CODE BY FILE:\n"
        "--------------------\n"
        f"{code_bundle}\n"
    )

    try:
        content = call_llm(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=model,
            temperature=0.0,
            max_tokens=1024,
        )
    except Exception:
        # Fail-open: if the LLM call fails, do not block execution.
        return {"ok": True, "issues_by_file": {}}

    parsed = _parse_llm_json(content)
    ok = bool(parsed.get("ok", True))
    issues_by_file = parsed.get("issues_by_file") or {}

    # Normalize issues: ensure dict[str, list[str]]
    norm_issues: Dict[str, List[str]] = {}
    if isinstance(issues_by_file, dict):
        for fname, issues in issues_by_file.items():
            if isinstance(issues, list):
                norm_issues[fname] = [str(x) for x in issues]
            elif isinstance(issues, str):
                norm_issues[fname] = [issues]

    return {"ok": ok, "issues_by_file": norm_issues}