from __future__ import annotations

import json
import re
from typing import Any, Dict

from ..utils.llm_client import call_llm


def _strip_markdown_fences(text: str) -> str:
    """
    Remove leading/trailing ``` or ```json fences if present.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # Drop first fence line
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    # Drop last fence line if it is also a fence
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _manual_single_file_parse(cleaned: str) -> Dict[str, Any]:
    """
    Very forgiving parser for responses of the form:

    {
      "entrypoint": "main.py",
      "code_by_file": {
        "main.py": "....arbitrary python with newlines and \"escaped\" quotes...."
      }
    }

    This does NOT require legal JSON (e.g., unescaped newlines inside the code
    string are tolerated). It only assumes:
      - double quotes delimit keys and the outer string,
      - inner quotes in code are escaped as \" (as in your example).
    """
    # entrypoint
    entrypoint = "main.py"
    m_ep = re.search(r'"entrypoint"\s*:\s*"([^"]+)"', cleaned)
    if m_ep:
        entrypoint = m_ep.group(1)

    # locate "code_by_file": {
    m_cb = re.search(r'"code_by_file"\s*:\s*\{', cleaned)
    if not m_cb:
        raise ValueError(
            "Could not find 'code_by_file' block in LLM response during manual parse."
        )
    idx = m_cb.end()

    # Skip whitespace, expect opening quote for filename
    n = len(cleaned)
    while idx < n and cleaned[idx].isspace():
        idx += 1
    if idx >= n or cleaned[idx] != '"':
        raise ValueError(
            "Manual parse expected '\"' starting filename in code_by_file block."
        )
    idx += 1
    fname_start = idx
    while idx < n and cleaned[idx] != '"':
        idx += 1
    if idx >= n:
        raise ValueError("Manual parse could not find end of filename string.")
    filename = cleaned[fname_start:idx]

    # Move to code string: find ':' then the opening quote of the code
    idx = cleaned.find(":", idx)
    if idx == -1:
        raise ValueError("Manual parse could not find ':' after filename.")
    idx += 1
    while idx < n and cleaned[idx].isspace():
        idx += 1
    if idx >= n or cleaned[idx] != '"':
        raise ValueError("Manual parse expected '\"' starting code string.")
    idx += 1
    code_start = idx

    # Scan until closing unescaped double quote
    escaped = False
    while idx < n:
        ch = cleaned[idx]
        if ch == "\\" and not escaped:
            escaped = True
            idx += 1
            continue
        if ch == '"' and not escaped:
            break
        escaped = False
        idx += 1
    if idx >= n:
        raise ValueError("Manual parse could not find end of code string.")
    code_raw = cleaned[code_start:idx]

    # Unescape common sequences
    # Note: code_raw may already contain real newlines (from the LLM),
    # and possibly escaped \" and \\n forms.
    code = (
        code_raw.replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\"", '"')
        .replace("\\\\", "\\")
    )

    return {
        "entrypoint": entrypoint,
        "code_by_file": {filename: code},
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    """
    Try very hard to turn the LLM response into a Python dict.

    Strategy:
      1. Strip markdown fences.
      2. Try json.loads on the whole string.
      3. Try json.loads on the { ... } slice between first and last brace.
      4. If all JSON parsing fails, fall back to a manual single-file parser
         that understands the specific shape we asked the model for.
    """
    cleaned = _strip_markdown_fences(text)

    # direct attempt
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # attempt on first-{ ... last-}
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = cleaned[start : end + 1]
        try:
            obj = json.loads(snippet)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # Keep snippet for manual parse
            cleaned = snippet

    # manual fallback for the common single-file case
    return _manual_single_file_parse(cleaned)

def generate_broken_project_code(config: Any, CodeFile: str) -> str:
    """
    Call the coder model once and ask it to emit a JSON object of the form:

        {
          "entrypoint": "main.py",
          "code_by_file": {
            "main.py": "<complete python script>",
            "...": "..."
          }
        }

    The function returns only the `code_by_file` mapping.
    Your updated agent.py already normalizes this into the shape it needs.
    """
    # Choose a model name from the AgentConfig, with fallbacks
    model_name = getattr(config, "coder_model", None) or getattr(
        config, "llm_model", "openai/gpt-oss-120b"
    )
    system_prompt = (
        "You are a precise Python code transformer.\n\n"
        "Your job is to take Python code provided by the user and insert realistic errors.\n"
        "These errors may include (but are not limited to):\n"
        "- logic errors\n"
        "- incorrect variable names\n"
        "- broken control flow\n"
        "- missing imports\n"
        "- invalid assumptions\n\n"
        "Rules:\n"
        "- Return ONLY the modified Python code as a single string.\n"
        "- Do NOT wrap it in JSON.\n"
        "- Do NOT explain the changes.\n"
        "- Do NOT include commentary.\n"
        "- Preserve overall structure but introduce subtle or overt errors.\n"
    )

    user_prompt = (
        "Insert realistic errors into the following Python script and return ONLY the modified script:\n\n"
        f"{CodeFile}"
    )

    raw = call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
    )
    return raw

# def generate_irrelevant_code(config: Any, CodeFile: str) -> str:


def generate_incorrect_env(config: Any, env_file: str) -> str: 
    """
    Call the coder model once and ask it to emit a JSON object of the form:

        {
          "entrypoint": "main.py",
          "code_by_file": {
            "main.py": "<complete python script>",
            "...": "..."
          }
        }

    The function returns only the `code_by_file` mapping.
    Your updated agent.py already normalizes this into the shape it needs.
    """
    # Choose a model name from the AgentConfig, with fallbacks
    model_name = getattr(config, "coder_model", None) or getattr(
        config, "llm_model", "openai/gpt-oss-120b"
    )
    system_prompt = (
        "You are a precise environment.yaml transformer.\n\n"
        "Your job is to take a conda environment.yaml file provided by the user and insert "
        "realistic errors.\n\n"
        "These errors may include (but are not limited to):\n"
        "- incorrect or missing dependency versions\n"
        "- invalid or misspelled package names\n"
        "- malformed YAML structure\n"
        "- wrong indentation\n"
        "- missing required sections\n"
        "- impossible dependency constraints\n"
        "- wrong channel names\n\n"
        "Rules:\n"
        "- Return ONLY the modified environment.yaml text as a single string.\n"
        "- Do NOT wrap it in JSON.\n"
        "- Do NOT explain the changes.\n"
        "- Do NOT include commentary.\n"
        "- Preserve the overall structure but introduce subtle or overt errors.\n"
    )

    user_prompt = (
        "Insert realistic errors into the following conda environment.yaml specification "
        "and return ONLY the modified environment.yaml text:\n\n"
        f"{env_file}"
    )

    raw = call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
    )
    return raw

def generate_project_code(
    question: str,
    project_plan: Dict[str, Any],
    config: Any,
) -> Dict[str, str]:
    """
    Call the coder model once and ask it to emit a JSON object of the form:

        {
          "entrypoint": "main.py",
          "code_by_file": {
            "main.py": "<complete python script>",
            "...": "..."
          }
        }

    The function returns only the `code_by_file` mapping.
    Your updated agent.py already normalizes this into the shape it needs.
    """
    # Choose a model name from the AgentConfig, with fallbacks
    model_name = getattr(config, "coder_model", None) or getattr(
        config, "llm_model", "openai/gpt-oss-120b"
    )

    system_prompt = (
        "You are a precise Python 3 code generator.\n\n"
        "You MUST respond with a single JSON object of the form:\n"
        "{\n"
        '  \"entrypoint\": \"<one of the filenames from project_plan[\\\"files\\\"]>\",\n'
        '  \"code_by_file\": {\n'
        '    \"<filename1>\": \"<complete Python source for that file>\",\n'
        '    \"<filename2>\": \"<complete Python source for that file>\",\n'
        "    ...\n"
        "  }\n"
        "}\n\n"
        "Constraints:\n"
        "- Every key in code_by_file MUST be one of the filenames in project_plan['files'].\n"
        "- Each value in code_by_file MUST be the full, standalone Python code for that file.\n"
        "- Every value in code_by_file MUST also be a valid JSON string: escape newlines as \\n\n"
        "  and quotes as \\\". Do NOT insert raw unescaped newlines inside the string.\n"
        "- For a file named 'environment.yaml', write a valid conda environment file\n"
        "  with pinned Python version and the dependencies required by the project code,\n"
        "  but still encode it as a single JSON string with \\n for line breaks.\n"
        "- Do NOT include any natural-language commentary.\n"
        "- You MAY wrap the JSON in ```json fences; this will be stripped.\n"
    )

    user_prompt = (
        "QUESTION:\n"
        f"{question}\n\n"
        "PROJECT PLAN (JSON):\n"
        f"{json.dumps(project_plan, indent=2)}\n\n"
        "Implement the entire project described by this plan.\n"
        "Use matplotlib for any requested plots, and save them with the exact filenames\n"
        "given in project_plan['experiments'][*]['outputs']['plots'][*]['filename'].\n"
        "Ensure the entrypoint script prints the final numeric answer with:\n"
        "    print('Answer:', value)\n\n"
        "If 'environment.yaml' is one of the files, define a conda-style environment like:\n"
        "  name: agentic-paper\n"
        "  channels:\n"
        "    - conda-forge\n"
        "  dependencies:\n"
        "    - python=3.11\n"
        "    - numpy\n"
        "    - matplotlib\n"
        "and include any other libraries you actually import in the project code.\n\n"
        "Respond ONLY with the JSON object described above."
    )

    raw = call_llm(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model=model_name,
    )
    
    parsed = _extract_json_object(raw)

    code_by_file = parsed.get("code_by_file") or {}

    # Normalize to Dict[str, str]
    norm: Dict[str, str] = {}
    for k, v in code_by_file.items():
        norm[str(k)] = str(v)

    # returns dict with main.py and environment.yaml
    return norm