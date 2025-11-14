from __future__ import annotations

import json
from typing import Dict, Any, List, Optional

from openai import OpenAI
from ..utils.inference_auth_token import get_access_token


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=get_access_token(),
        base_url="https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    )


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Try to parse a JSON object from raw LLM text.

    - First try direct json.loads.
    - If that fails, look for the first '{' ... last '}' span and try json.loads on it.
    - On failure, return a minimal default: {"experiments": []}.
    """
    text = text.strip()

    # Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Strip markdown code fences if present
    if text.startswith("```"):
        # remove the first line (``` or ```json) and the last line (```)
        lines = text.splitlines()
        if len(lines) >= 2:
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            inner = "\n".join(lines).strip()
            try:
                obj = json.loads(inner)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                text = inner  # fall through to braces search with stripped text

    # Heuristic: find first '{' and last '}', parse substring
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass

    # Fallback
    return {"experiments": []}


def plan_experiments(
    question: str,
    model: Optional[str] = None,
    max_experiments: int = 2,
) -> Dict[str, Any]:
    """
    Ask an LLM to propose a small set of experiments for the given question.

    Args:
        question: Natural-language problem statement.
        model:    Model name for the planner; if None, use a default.
        max_experiments: Maximum number of experiments to keep.

    Returns:
        {
          "experiments": [
             {
               "id": "exp1",
               "name": "...",
               "goal": "...",
               "description": "...",
               "outputs": {
                 "plots": [
                   {"filename": "foo.png", "description": "..."},
                   ...
                 ],
                 "metrics": [
                   {"name": "final_error", "description": "..."},
                   ...
                 ]
               }
             },
             ...
          ]
        }
    """
    if model is None:
        model = "openai/gpt-oss-120b"

    system_prompt = f"""
You are an experiment planner for numerical and scientific computing projects.

Given a research question, you must propose a SMALL set of well-structured
experiments (at most {max_experiments}) that together answer the question.

Each experiment should have:
- a short ID (like "exp1", "exp2", ...),
- a concise name,
- a one-sentence goal,
- a 1–3 sentence description,
- a list of output plots with exact filenames,
- a list of quantitative metrics with exact names.

You MUST respond with a SINGLE JSON object of the form:

{{
  "experiments": [
    {{
      "id": "exp1",
      "name": "Short experiment title",
      "goal": "One-sentence goal of the experiment.",
      "description": "2–3 sentences describing what this experiment does.",
      "outputs": {{
        "plots": [
          {{
            "filename": "example_plot.png",
            "description": "What this plot shows."
          }}
        ],
        "metrics": [
          {{
            "name": "final_approx",
            "description": "What this metric measures."
          }}
        ]
      }}
    }}
  ]
}}

Constraints:
- The top-level key MUST be "experiments".
- You MUST provide between 1 and {max_experiments} experiments.
- Do NOT include any comments, prose, or markdown outside the JSON.
- Do NOT invent any other top-level keys.
"""

    user_prompt = (
        "Research question:\n"
        "------------------\n"
        f"{question}\n\n"
        "Design a small, coherent set of experiments that, if implemented in Python,\n"
        "would answer this question. Use only JSON in your reply, following the\n"
        "schema described in the system message."
    )

    client = _get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    raw = resp.choices[0].message.content or ""
    parsed = _extract_json_object(raw)

    experiments = parsed.get("experiments") or []
    if not isinstance(experiments, list):
        experiments = []

    # Clip to max_experiments if the model produced more
    if len(experiments) > max_experiments:
        experiments = experiments[:max_experiments]

    return {"experiments": experiments}