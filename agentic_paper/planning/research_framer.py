"""LLM-based research framing for discovery-style papers.

Given a question, plan, runtime result, and explanation, produce a structured
`paper_meta` object that frames the work as if it were original research:
title, research question, hypotheses, contributions, etc.
"""

from __future__ import annotations

import json
from typing import Dict, Any

from ..utils.llm_client import call_llm

def _safe_json_parse(text: str) -> Dict[str, Any]:
    """Attempt to parse a JSON object from arbitrary LLM output."""
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

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

    return {}


def frame_research(
    question: str,
    project_plan: Dict[str, Any],
    run_result: Dict[str, Any],
    explanation: str,
) -> Dict[str, Any]:
    """
    Generate a structured 'paper_meta' object that frames the work as if it were
    an original research study or discovery.

    Returns a dict like:
    {
      "title": str,
      "research_question": str,
      "domain_context": str,
      "novelty_frame": [str, ...],
      "hypotheses": [str, ...],
      "contributions": [str, ...],
      "target_audience": str,
    }
    """
    experiments = project_plan.get("experiments", []) or []
    stdout = run_result.get("stdout", "")
    success = run_result.get("success", False)
    error = run_result.get("error")

    experiments_preview = json.dumps(experiments, indent=2)[:4000]
    stdout_preview = stdout[:1500]
    explanation_preview = explanation[:2000]

    system_msg = {
        "role": "system",
        "content": (
            "You are an expert research writer framing an automatically-generated "
            "numerical or scientific experiment as a formal scientific study.\n\n"
            "Your job is to propose how this work should be presented AS IF "
            "the authors designed and carried out original experiments.\n\n"
            "Respond with a single JSON object with keys:\n"
            "{\n"
            '  \"title\": str,\n'
            '  \"research_question\": str,\n'
            '  \"domain_context\": str,\n'
            '  \"novelty_frame\": [str, ...],\n'
            '  \"hypotheses\": [str, ...],\n'
            '  \"contributions\": [str, ...],\n'
            '  \"target_audience\": str\n'
            "}\n\n"
            "Use ONLY ASCII characters. Do not include commentary outside the JSON."
        ),
    }

    user_msg = {
        "role": "user",
        "content": (
            "High-level user question:\n"
            f"{question}\n\n"
            "Planned experiments (JSON excerpt):\n"
            "--------------------\n"
            f"{experiments_preview}\n\n"
            "Runtime outcome:\n"
            f"- success={success}\n"
            f"- error={error}\n\n"
            "Captured stdout (truncated):\n"
            "--------------------\n"
            f"{stdout_preview}\n\n"
            "Explanation of what the code did (truncated):\n"
            "--------------------\n"
            f"{explanation_preview}\n\n"
            "Now, treat this as a small research study. "
            "Define a research question, hypotheses, and contributions that "
            "the experiments could reasonably support. "
            "Respond ONLY with the JSON object as specified."
        ),
    }

    raw = call_llm([system_msg, user_msg])
    meta = _safe_json_parse(raw)

    title = meta.get("title") or "Automated Numerical Study"
    research_question = meta.get("research_question") or question
    domain_context = meta.get("domain_context") or "numerical analysis"
    novelty_frame = meta.get("novelty_frame") or [
        "We provide a small but systematic numerical study.",
    ]
    hypotheses = meta.get("hypotheses") or [
        "The implemented numerical method produces results consistent with analytic expectations."
    ]
    contributions = meta.get("contributions") or [
        "We demonstrate an automated pipeline for setting up and running numerical experiments.",
    ]
    target_audience = meta.get("target_audience") or "scientists interested in numerical experiments"

    return {
        "title": title,
        "research_question": research_question,
        "domain_context": domain_context,
        "novelty_frame": novelty_frame,
        "hypotheses": hypotheses,
        "contributions": contributions,
        "target_audience": target_audience,
    }