from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from ..utils.llm_client import call_llm


def _safe_parse_json_object(text: str) -> Dict[str, Any]:
    """
    Try to parse a JSON object from an LLM response.

    - Prefer direct json.loads.
    - Fallback: use the substring between the first '{' and the last '}'.
    - On failure, return a minimal default plan with no experiments.
    """
    text = text.strip()

    # 1) Direct parse
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # 2) Try to extract a JSON object substring
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

    # 3) Fallback
    return {"experiments": []}


def plan_experiments(
    question: str,
    model: Optional[str] = None,
    max_experiments: int = 3,
) -> Dict[str, Any]:
    """
    Ask an LLM to propose a small set of experiments for the given question.

    Simplified schema (NO metrics block, to reduce brittleness):

    {
      "experiments": [
        {
          "id": "exp1",
          "name": "<short descriptive name>",
          "goal": "<one-sentence goal>",
          "description": "<2–5 sentence description>",
          "outputs": {
            "plots": [
              {
                "filename": "displacement_vs_time.png",
                "description": "Displacement as a function of time."
              },
              ...
            ]
          }
        },
        ...
      ]
    }

    The filenames are only used to enforce that the code calls plt.savefig(...)
    with those exact strings; we do NOT require any particular metric variable
    names in the code anymore.
    """
    if model is None:
        # Use whatever default planner model you like; you can override via config
        model = "openai/gpt-4o-mini"

    # Describe the target JSON format in plain text (no f-string interpolation
    # inside the JSON example, to avoid format-specifier issues).
    schema_description = (
        "You must respond with a SINGLE JSON object of the form:\n"
        "{\n"
        '  "experiments": [\n'
        '    {\n'
        '      "id": "exp1",\n'
        '      "name": "Short experiment name",\n'
        '      "goal": "One-sentence goal of the experiment.",\n'
        '      "description": "2-5 sentences describing what will be done.",\n'
        '      "outputs": {\n'
        '        "plots": [\n'
        '          {\n'
        '            "filename": "some_plot_name.png",\n'
        '            "description": "What this plot shows."\n'
        '          }\n'
        '        ]\n'
        '      }\n'
        '    }\n'
        '  ]\n'
        "}\n"
    )

    system_prompt = (
        "You are an experimental design assistant for numerical / scientific Python projects.\n"
        "Your job is to propose a SMALL set of well-structured experiments for a coding agent.\n\n"
        "Constraints:\n"
        "- Focus on 1 to 3 experiments that are genuinely useful to answer the question.\n"
        "- Each experiment should have a clear goal and a concise description.\n"
        "- For each experiment, specify one or more PNG plots to generate using matplotlib.\n"
        "- Use simple snake_case filenames ending in .png (e.g., 'displacement_vs_time.png').\n"
        "- Do NOT include any 'metrics' array or variable names; only describe plots.\n"
        "- Do NOT include any commentary outside the JSON object.\n\n"
        + schema_description
        + "\n"
        f"When designing experiments, think in terms of what code a Python script can do: "
        f"simulate, sweep parameters, and produce plots.\n"
    )

    user_prompt = (
        "QUESTION:\n"
        f"{question}\n\n"
        "Design experiments that would help solve or illustrate this question.\n"
        "Remember: respond with ONLY a single JSON object matching the schema described.\n"
    )

    raw = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    parsed = _safe_parse_json_object(raw)
    experiments: List[Dict[str, Any]] = []

    raw_exps = parsed.get("experiments", [])
    if isinstance(raw_exps, list):
        for idx, exp in enumerate(raw_exps):
            if not isinstance(exp, dict):
                continue
            # Ensure minimal keys are present
            exp_id = exp.get("id") or f"exp{idx + 1}"
            name = exp.get("name") or f"Experiment {idx + 1}"
            goal = exp.get("goal") or ""
            description = exp.get("description") or ""
            outputs = exp.get("outputs") or {}
            plots = outputs.get("plots") or []

            # Normalize plots to list[dict]
            norm_plots: List[Dict[str, Any]] = []
            if isinstance(plots, list):
                for p in plots:
                    if not isinstance(p, dict):
                        continue
                    fn = p.get("filename")
                    desc = p.get("description") or ""
                    if not fn or not isinstance(fn, str):
                        continue
                    norm_plots.append(
                        {
                            "filename": fn,
                            "description": desc,
                        }
                    )

            experiments.append(
                {
                    "id": exp_id,
                    "name": name,
                    "goal": goal,
                    "description": description,
                    "outputs": {
                        "plots": norm_plots,
                        # No metrics field on purpose; the validator will then
                        # have nothing metric-specific to enforce.
                    },
                }
            )

    # Clip to max_experiments
    if max_experiments is not None and max_experiments > 0:
        experiments = experiments[:max_experiments]

    return {"experiments": experiments}