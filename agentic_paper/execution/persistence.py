from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from agentic_paper.config import AgentConfig

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
no_code_saved_error = AgentConfig.no_code_saved
no_env_saved_error = AgentConfig.no_env_saved

def _slugify(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    text = _SLUG_RE.sub("_", text).strip("_")
    if len(text) > max_len:
        text = text[:max_len].rstrip("_")
    return text or "run"


def create_experiment_dirs(base_dir: str, question: str) -> Dict[str, str]:
    """
    Create a unique, timestamped directory for this run and useful subfolders.

    Layout:

        base_dir/
          20251114_134620_numerically_approximate_the_definite_integral/
            code/
            paper/
            markdown/
            figures/
            state.json
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(question)

    root = base / f"{ts}_{slug}"
    paper_dir = root / "paper"
    markdown_dir = root / "markdown"
    figures_dir = root / "figures"

    for d in (root, paper_dir, markdown_dir, figures_dir):
        d.mkdir(parents=True, exist_ok=True)

    if not no_code_saved_error:
        code_dir = root / "code"
        code_dir.mkdir(parents=True, exist_ok=True)
        return {
        "root_dir": str(root),
        "code_dir": str(code_dir),
        "paper_dir": str(paper_dir),
        "markdown_dir": str(markdown_dir),
        "figures_dir": str(figures_dir),
        }
    else: 
        return {
        "root_dir": str(root),
        "paper_dir": str(paper_dir),
        "markdown_dir": str(markdown_dir),
        "figures_dir": str(figures_dir),
        }


def save_experiment_artifacts(
    experiment_dirs: Dict[str, str],
    question: str,
    project_plan: Dict[str, Any],
    code_by_file: Dict[str, str],
    combined_code: str,
    run_result: Dict[str, Any],
    explanation: str,
    parsed_answer: Dict[str, Any],
    critic_report: Dict[str, Any],
    methods_text: str,
    background_text: str,
    results_text: str,
    introduction_text: str,
    discussion_text: str,
    paper_tex: str,
    references_bib: str,
    attempts_meta: List[Dict[str, Any]],
    related_work_text: str,
    repo_info: Dict[str, Any] | None = None,
) -> Dict[str, str]:
    """
    Persist all artifacts into the per-run directory.
    """
    root = Path(experiment_dirs["root_dir"])
    paper_dir = Path(experiment_dirs["paper_dir"])
    markdown_dir = Path(experiment_dirs["markdown_dir"])
    figures_dir = Path(experiment_dirs["figures_dir"])

    # Markdown sections
    (markdown_dir / "introduction.md").write_text(introduction_text, encoding="utf-8")
    (markdown_dir / "related_work.md").write_text(related_work_text, encoding="utf-8")
    (markdown_dir / "methods.md").write_text(methods_text, encoding="utf-8")
    (markdown_dir / "background.md").write_text(background_text, encoding="utf-8")
    (markdown_dir / "results.md").write_text(results_text, encoding="utf-8")
    (markdown_dir / "discussion.md").write_text(discussion_text, encoding="utf-8")
    (markdown_dir / "explanation.md").write_text(explanation, encoding="utf-8")

    # LaTeX + BibTeX 
    (paper_dir / "paper.tex").write_text(paper_tex, encoding="utf-8")
    (paper_dir / "references.bib").write_text(references_bib, encoding="utf-8")

    # Metadata / state log 
    meta = {
        "question": question,
        "project_plan": project_plan,
        "run_result": {k: v for k, v in run_result.items() if k != "locals"},
        "parsed_answer": parsed_answer,
        "critic_report": critic_report,
        "attempts": attempts_meta,
    }
    if repo_info is not None:
        meta["repo"] = repo_info
    state_path = root / "state.json"
    state_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Figures
    for png in root.glob("*.png"):
        dest = figures_dir / png.name
        print("dest: ", dest)
        try:
            shutil.move(str(png), dest)
        except Exception:
            # If something goes wrong, leave the file where it is.
            print("figure could not be moved!!!")
            pass

    print("figures_dir: ", figures_dir)

    result = {
        "root": str(root),
        "paper_dir": str(paper_dir),
        "markdown_dir": str(markdown_dir),
        "figures_dir": str(figures_dir),
        "state_json": str(state_path),
        "paper_tex": str(paper_dir / "paper.tex"),
        "references_bib": str(paper_dir / "references.bib"),
        "repo_url": (repo_info or {}).get("repo", {}).get("html_url") if repo_info else None,
        }

    # Code
    if not no_code_saved_error:
        code_dir = Path(experiment_dirs["code_dir"])
        for name, code in code_by_file.items():
            if name =="environment.yaml" and no_env_saved_error: 
                continue
            (code_dir / name).write_text(code, encoding="utf-8")
        (code_dir / "combined_code.py").write_text(combined_code, encoding="utf-8")
        result["code_dir"] = str(code_dir)
    return result
