"""High-level coding agent API with project planning, codegen, validation,
retry logic, execution, explanation, and paper generation.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List, Optional
from pathlib import Path

from .config import AgentConfig
from .planning.project_planner import plan_project
from .codegen.codegen import generate_project_code
from .execution.runner import run_generated_code
from .execution.explainer import generate_explanation
from .execution.persistence import (
    create_experiment_dirs,
    save_experiment_artifacts,
)
from .validation.validator import validate_code
from .codegen.code_editor import revise_code
from .execution.answer_parser import parse_answer_from_stdout
from .validation.critic import simple_critic
from .validation.spec_critic import spec_critic

from .writing.methods_writer import write_methods
from .writing.results_writer import write_results
from .writing.introduction_writer import write_introduction
from .writing.background_writer import write_background
from .writing.related_work_writer import write_related_work
from .writing.discussion_writer import write_discussion
from .writing.paper_writer import assemble_acm_paper_tex
from .writing.reference_writer import generate_references_bib
from .writing.paper_polisher import polish_paper_tex
from .writing.latex_sanitize import sanitize_latex


# Internal helpers

_FUNC_PATTERN = re.compile(
    r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(",
    re.MULTILINE,
)


def _combine_project_code(
    project_plan: Dict[str, Any],
    code_by_file: Dict[str, str],
) -> str:
    """Combine project files into a single script string for execution/logging."""
    parts: List[str] = []
    for file_spec in project_plan.get("files", []):
        name = file_spec["name"]
        role = file_spec.get("role")
        if role == "env":
            continue  # skip environment.yaml
        code = code_by_file.get(name, "")
        parts.append(f"# ==== {name} ====")
        parts.append(code)
        parts.append("")  # blank line
    return "\n".join(parts)


def _extract_function_names_from_code(code: str) -> List[str]:
    """Extract top-level function names from a Python source string."""
    return _FUNC_PATTERN.findall(code)


def _get_helper_function_names(
    project_plan: Dict[str, Any],
    code_by_file: Dict[str, str],
    entrypoint_name: Optional[str],
) -> List[str]:
    """Collect function names defined in non-entrypoint files (helpers)."""
    names: List[str] = []
    for f in project_plan.get("files", []):
        name = f["name"]
        if entrypoint_name is not None and name == entrypoint_name:
            continue
        code = code_by_file.get(name, "")
        names.extend(_extract_function_names_from_code(code))

    # Deduplicate while preserving order
    seen = set()
    unique_names: List[str] = []
    for n in names:
        if n not in seen:
            seen.add(n)
            unique_names.append(n)
    return unique_names


def _entrypoint_uses_helpers(code: str, helper_function_names: List[str]) -> bool:
    """Check whether entrypoint code calls at least one helper function.

    Do not count function definitions as usage; only call sites.
    """
    if not helper_function_names:
        return False

    lines = code.splitlines()
    for line in lines:
        stripped = line.lstrip()
        if stripped.startswith("def ") or stripped.startswith("class "):
            continue
        for fn in helper_function_names:
            pattern = r"\b" + re.escape(fn) + r"\s*\("
            if re.search(pattern, line):
                return True
    return False


def _expected_plot_filenames(project_plan: Dict[str, Any]) -> List[str]:
    """Collect all expected plot filenames from the experiment plan."""
    filenames: List[str] = []
    for exp in project_plan.get("experiments", []) or []:
        outputs = exp.get("outputs") or {}
        for p in outputs.get("plots", []) or []:
            fn = p.get("filename")
            if fn:
                filenames.append(fn)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for fn in filenames:
        if fn not in seen:
            seen.add(fn)
            unique.append(fn)
    return unique


def _entrypoint_generates_required_plots(
    combined_code: str,
    project_plan: Dict[str, Any],
) -> List[str]:
    """Return list of missing expected plot filenames.

    We simply check that every expected PNG filename from the experiment plan
    appears somewhere in the combined project code (usually in plt.savefig()).
    """
    expected = _expected_plot_filenames(project_plan)
    if not expected:
        return []

    missing: List[str] = []
    for fn in expected:
        if fn not in combined_code:
            missing.append(fn)
    return missing


# Main agent API

def solve_question_with_agent(
    question: str,
    config: AgentConfig,
) -> Dict[str, Any]:
    if config is None:
        config = AgentConfig()

    # Create per-run directory structure
    experiment_dirs = create_experiment_dirs(config.base_dir, question)

    # Plan the project
    project_plan = plan_project(question, config=config)

    # Generate initial code for all files
    gen_result = generate_project_code(question, project_plan, config=config)

    entrypoint_name: Optional[str] = None  # may be filled from gen_result or inferred from project_plan

    if isinstance(gen_result, dict):
        if "code_by_file" in gen_result:
            # New structured interface
            code_by_file: Dict[str, str] = gen_result["code_by_file"]
            entrypoint_name = gen_result.get("entrypoint")
        else:
            # Plain mapping filename -> code
            code_by_file = gen_result  # type: ignore[assignment]
    else:
        # Tuple / list interface
        if not gen_result:
            raise ValueError("generate_project_code returned an empty result")
        code_by_file = gen_result[0]  # type: ignore[index]

    # If codegen did not tell us the entrypoint, infer it from the plan
    if entrypoint_name is None:
        for f in project_plan.get("files", []):
            if f.get("role") == "entrypoint":
                entrypoint_name = f["name"]
                break
        # Fallback: first file in the plan
        if entrypoint_name is None:
            files = project_plan.get("files") or []
            if files:
                entrypoint_name = files[0]["name"]

    attempts_meta: List[Dict[str, Any]] = []
    final_run_result: Dict[str, Any] = {
        "success": False,
        "stdout": "",
        "locals": {},
        "error": None,
    }

    total_attempts = 1 + max(getattr(config, "max_retries", 0), 0)

    # Main attempt + retries loop
    for attempt_idx in range(total_attempts):
        combined_code_for_validation = _combine_project_code(
            project_plan, code_by_file
        )

        # Spec critic (static consistency; may be a no-op)
        try:
            spec_result = spec_critic(
                project_plan=project_plan,
                code_by_file=code_by_file,
            )
            spec_ok = bool(spec_result.get("ok", True))
            spec_issues_by_file = spec_result.get("issues_by_file", {}) or {}
        except Exception:
            spec_ok = True
            spec_issues_by_file = {}

        all_valid = spec_ok
        validation_issues_by_file: Dict[str, List[str]] = {
            fname: list(issues)
            for fname, issues in spec_issues_by_file.items()
        }

        # Per-file validation
        for f in project_plan.get("files", []):
            name = f["name"]
            role = f.get("role")
            code = code_by_file.get(name, "")

            require_answer = role == "entrypoint"
            is_valid, issues = validate_code(
                code=code,
                config=config,
                require_answer_print=require_answer,
            )
            if not is_valid:
                all_valid = False
                if issues:
                    validation_issues_by_file.setdefault(name, []).extend(issues)

        # Optional: ensure entrypoint uses helpers (only if multiple files)
        helper_function_names = _get_helper_function_names(
            project_plan, code_by_file, entrypoint_name
        )
        if helper_function_names and entrypoint_name is not None:
            entry_code = code_by_file.get(entrypoint_name, "")
            if not _entrypoint_uses_helpers(entry_code, helper_function_names):
                all_valid = False
                msg = (
                    "Entrypoint file should call at least one helper function; "
                    "expected to use one of: "
                    + ", ".join(sorted(helper_function_names))
                )
                validation_issues_by_file.setdefault(entrypoint_name, []).append(msg)

        # Ensure code references all required plots
        missing_plots = _entrypoint_generates_required_plots(
            combined_code_for_validation, project_plan
        )
        if missing_plots:
            all_valid = False
            # Attach the error to the entrypoint if we have one; otherwise to a
            # generic synthetic file name.
            target_file = entrypoint_name or "main.py"
            msg = (
                "The project plan defines experiments with plots, but the code "
                "does not reference all expected PNG filenames. "
                "Ensure each of these appears in a plt.savefig(...) call: "
                + ", ".join(missing_plots)
            )
            validation_issues_by_file.setdefault(target_file, []).append(msg)

        # Write environment.yaml file to be used when running code
        root_dir = experiment_dirs["root_dir"]
        env_code = code_by_file.get("environment.yaml")
        if env_code:
            env_path = Path(root_dir) / "environment.yaml"
            env_path.write_text(env_code, encoding="utf-8")

        # Execute if validation passed
        if all_valid:
            combined_code = combined_code_for_validation
            final_run_result = run_generated_code(
                combined_code,
                work_dir=experiment_dirs["root_dir"],
                entrypoint_name=entrypoint_name or "main.py",
                env_tool="conda",
                use_isolated_env=True,
            )
        else:
            combined_code = combined_code_for_validation
            final_run_result = {
                "success": False,
                "stdout": "",
                "locals": {},
                "error": f"ValidationError: {validation_issues_by_file}",
            }

        attempts_meta.append(
            {
                "attempt_index": attempt_idx,
                "success": final_run_result.get("success", False),
                "error": final_run_result.get("error"),
                "validation_issues": validation_issues_by_file,
            }
        )

        # Stop if successful and valid
        if final_run_result.get("success") and all_valid:
            break

        # If retries remain, revise the entrypoint code only
        if attempt_idx < total_attempts - 1 and entrypoint_name is not None:
            entry_code = code_by_file.get(entrypoint_name, "")
            issues_for_revision: List[str] = []
            if validation_issues_by_file:
                for iss_list in validation_issues_by_file.values():
                    issues_for_revision.extend(iss_list)

            code_by_file[entrypoint_name] = revise_code(
                question=question,
                previous_code=entry_code,
                run_result=final_run_result,
                validation_issues=issues_for_revision,
            )
        else:
            break

    # Final combined code (for explanation / persistence)
    combined_code = _combine_project_code(project_plan, code_by_file)

    # Parse numeric answer from stdout
    parsed_answer = parse_answer_from_stdout(final_run_result.get("stdout", ""))

    # Natural-language explanation
    explanation = generate_explanation(question, combined_code, final_run_result)

    # Critic pass
    critic_report = simple_critic(
        question=question,
        parsed_answer=parsed_answer,
        explanation=explanation,
        run_result=final_run_result,
    )

    # Section writers (Markdown)
    methods_text = write_methods(
        question=question,
        project_plan=project_plan,
        code_by_file=code_by_file,
        parsed_answer=parsed_answer,
        run_result=final_run_result,
    )

    results_text = write_results(
        question=question,
        parsed_answer=parsed_answer,
        run_result=final_run_result,
        critic_report=critic_report,
        explanation=explanation,
    )

    introduction_text = write_introduction(
        question=question,
        methods_text=methods_text,
        results_text=results_text,
    )

    background_text = write_background(
        question=question,
        introduction_text=introduction_text,
        methods_text=methods_text,
        results_text=results_text,
    )

    related_work_text = write_related_work(
        question=question,
        introduction_text=introduction_text,
        background_text=background_text,
        methods_text=methods_text,
        results_text=results_text,
    )

    discussion_text = write_discussion(
        question=question,
        methods_text=methods_text,
        results_text=results_text,
        parsed_answer=parsed_answer,
        critic_report=critic_report,
    )

    # BibTeX references
    references_bib = generate_references_bib(
        question=question,
        introduction_text=introduction_text,
        methods_text=methods_text,
        results_text=results_text,
        discussion_text=discussion_text,
    )

    # Assemble ACM-style LaTeX paper
    paper_tex = assemble_acm_paper_tex(
        question=question,
        intro_md=introduction_text,
        background_md=background_text,
        related_work_md=related_work_text,
        methods_md=methods_text,
        results_md=results_text,
        discussion_md=discussion_text,
        project_plan=project_plan,
        references_bib=references_bib,
    )

    if paper_tex and getattr(config, "enable_paper_polish", False):
        try:
            paper_tex = polish_paper_tex(
                paper_tex,
                critique_model=config.editor_model,
                edit_model=config.editor_model,
            )
        except Exception:
            pass

    paper_tex = sanitize_latex(paper_tex)

    # Persist everything into the per-run directory
    paths = save_experiment_artifacts(
        experiment_dirs=experiment_dirs,
        question=question,
        project_plan=project_plan,
        code_by_file=code_by_file,
        combined_code=combined_code,
        run_result=final_run_result,
        explanation=explanation,
        parsed_answer=parsed_answer,
        critic_report=critic_report,
        methods_text=methods_text,
        results_text=results_text,
        introduction_text=introduction_text,
        background_text=background_text,
        discussion_text=discussion_text,
        paper_tex=paper_tex,
        references_bib=references_bib,
        attempts_meta=attempts_meta,
        related_work_text=related_work_text,
    )

    return {
        "question": question,
        "project_plan": project_plan,
        "code_by_file": code_by_file,
        "combined_code": combined_code,
        "run_result": final_run_result,
        "explanation": explanation,
        "parsed_answer": parsed_answer,
        "critic_report": critic_report,
        "sections": {
            "introduction": introduction_text,
            "related_work": related_work_text,
            "background": background_text,
            "methods": methods_text,
            "results": results_text,
            "discussion": discussion_text,
        },
        "paper_tex": paper_tex,
        "references_bib": references_bib,
        "paths": paths,
    }


if __name__ == "__main__":
    cfg = AgentConfig(base_dir="experiments", max_retries=0)
    q = (
        "Simulate a damped harmonic oscillator"
    )
    result = solve_question_with_agent(q, config=cfg)

    print("QUESTION:")
    print(result["question"])
    for k, v in result["paths"].items():
        print(f"{k}: {v}")