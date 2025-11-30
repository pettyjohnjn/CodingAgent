"""Academy-py agents that wrap the existing pipeline components.

Agents:
- PlanningAgent -> planning.project_planner.plan_project
- CodegenAgent -> codegen.codegen.generate_project_code / code_editor.revise_code
- ValidationAgent -> spec_critic + validate_code checks
- ExecutionAgent -> execution.runner.run_generated_code + helpers
- WritingAgent -> writing/* modules to assemble paper
- GithubAgent -> execution.github_publisher for repo + upload

The orchestrator (`solve_with_academy_agents`) launches these agents on a local
exchange and coordinates the same steps previously handled inline in agent.py.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from academy.agent import Agent, action
    from academy.exchange import LocalExchangeFactory
    from academy.manager import Manager
except ImportError as exc:  # pragma: no cover - clearer error when dependency missing
    raise ImportError(
        "academy-py is required. Install with `pip install academy-py`."
    ) from exc

from .agent import (
    _append_repo_note,
    _combine_project_code,
    _entrypoint_generates_required_plots,
    _entrypoint_uses_helpers,
    _get_helper_function_names,
)
from .codegen.codegen import generate_project_code
from .codegen.code_editor import revise_code
from .config import AgentConfig
from .execution.answer_parser import parse_answer_from_stdout
from .execution.explainer import generate_explanation
from .execution.github_publisher import create_repo, upload_run_artifacts
from .execution.persistence import create_experiment_dirs, save_experiment_artifacts
from .execution.runner import run_generated_code
from .planning.project_planner import plan_project
from .validation.critic import simple_critic
from .validation.spec_critic import spec_critic
from .validation.validator import validate_code
from .writing.background_writer import write_background
from .writing.discussion_writer import write_discussion
from .writing.introduction_writer import write_introduction
from .writing.methods_writer import write_methods
from .writing.paper_polisher import polish_paper_tex
from .writing.paper_writer import assemble_acm_paper_tex
from .writing.reference_writer import generate_references_bib
from .writing.related_work_writer import write_related_work
from .writing.results_writer import write_results
from .writing.latex_sanitize import sanitize_latex


# === Individual agents ===


class PlanningAgent(Agent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()

    @action
    async def plan(self, question: str) -> Dict[str, Any]:
        return await self.agent_run_sync(plan_project, question, self.config)


class CodegenAgent(Agent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()

    @action
    async def generate(self, question: str, project_plan: Dict[str, Any]) -> Dict[str, str]:
        return await self.agent_run_sync(generate_project_code, question, project_plan, self.config)

    @action
    async def revise(
        self,
        question: str,
        previous_code: str,
        run_result: Dict[str, Any],
        validation_issues: List[str],
    ) -> str:
        return await self.agent_run_sync(
            revise_code,
            question,
            previous_code,
            run_result,
            validation_issues,
        )


class ValidationAgent(Agent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()

    @action
    async def validate(
        self,
        project_plan: Dict[str, Any],
        code_by_file: Dict[str, str],
        entrypoint_name: Optional[str],
    ) -> Dict[str, Any]:
        def _validate_sync() -> Dict[str, Any]:
            try:
                spec_result = spec_critic(project_plan=project_plan, code_by_file=code_by_file)
                spec_ok = bool(spec_result.get("ok", True))
                spec_issues_by_file = spec_result.get("issues_by_file", {}) or {}
            except Exception:
                spec_ok = True
                spec_issues_by_file = {}

            all_valid = spec_ok
            validation_issues_by_file: Dict[str, List[str]] = {
                fname: list(issues) for fname, issues in spec_issues_by_file.items()
            }

            for f in project_plan.get("files", []):
                name = f["name"]
                role = f.get("role")
                code = code_by_file.get(name, "")
                require_answer = role == "entrypoint"
                is_valid, issues = validate_code(
                    code=code,
                    config=self.config,
                    require_answer_print=require_answer,
                )
                if not is_valid:
                    all_valid = False
                    if issues:
                        validation_issues_by_file.setdefault(name, []).extend(issues)

            helper_function_names = _get_helper_function_names(project_plan, code_by_file, entrypoint_name)
            if helper_function_names and entrypoint_name is not None:
                entry_code = code_by_file.get(entrypoint_name, "")
                if not _entrypoint_uses_helpers(entry_code, helper_function_names):
                    all_valid = False
                    msg = (
                        "Entrypoint file should call at least one helper function; "
                        "expected to use one of: " + ", ".join(sorted(helper_function_names))
                    )
                    validation_issues_by_file.setdefault(entrypoint_name, []).append(msg)

            combined_for_plots = _combine_project_code(project_plan, code_by_file)
            missing_plots = _entrypoint_generates_required_plots(combined_for_plots, project_plan)
            if missing_plots:
                all_valid = False
                target_file = entrypoint_name or "main.py"
                msg = (
                    "The project plan defines experiments with plots, but the code does not reference all expected "
                    "PNG filenames. Ensure each of these appears in a plt.savefig(...): "
                    + ", ".join(missing_plots)
                )
                validation_issues_by_file.setdefault(target_file, []).append(msg)

            return {
                "all_valid": all_valid,
                "issues_by_file": validation_issues_by_file,
            }

        return await self.agent_run_sync(_validate_sync)

    @action
    async def critic(
        self,
        question: str,
        parsed_answer: Any,
        explanation: str,
        run_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        return await self.agent_run_sync(
            simple_critic,
            question,
            parsed_answer,
            explanation,
            run_result,
        )


class ExecutionAgent(Agent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()

    @action
    async def run(
        self,
        combined_code: str,
        work_dir: str,
        entrypoint_name: str,
        env_tool: str,
        use_isolated_env: bool,
    ) -> Dict[str, Any]:
        return await self.agent_run_sync(
            run_generated_code,
            combined_code,
            work_dir,
            entrypoint_name,
            env_tool,
            use_isolated_env,
        )

    @action
    async def parse_answer(self, stdout: str) -> Any:
        return await self.agent_run_sync(parse_answer_from_stdout, stdout)

    @action
    async def explain(
        self,
        question: str,
        combined_code: str,
        run_result: Dict[str, Any],
    ) -> str:
        return await self.agent_run_sync(generate_explanation, question, combined_code, run_result)


class WritingAgent(Agent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()

    @action
    async def write_all(
        self,
        question: str,
        project_plan: Dict[str, Any],
        code_by_file: Dict[str, str],
        parsed_answer: Any,
        run_result: Dict[str, Any],
        critic_report: Dict[str, Any],
        explanation: str,
        repo_url: Optional[str],
    ) -> Dict[str, Any]:
        def _write_sync() -> Dict[str, Any]:
            methods_text = write_methods(
                question=question,
                project_plan=project_plan,
                code_by_file=code_by_file,
                parsed_answer=parsed_answer,
                run_result=run_result,
            )

            results_text = write_results(
                question=question,
                parsed_answer=parsed_answer,
                run_result=run_result,
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

            results_text_with_repo = _append_repo_note(results_text, repo_url)
            discussion_text_with_repo = _append_repo_note(discussion_text, repo_url)

            references_bib = generate_references_bib(
                question=question,
                introduction_text=introduction_text,
                methods_text=methods_text,
                results_text=results_text_with_repo,
                discussion_text=discussion_text_with_repo,
            )

            paper_tex = assemble_acm_paper_tex(
                question=question,
                intro_md=introduction_text,
                background_md=background_text,
                related_work_md=related_work_text,
                methods_md=methods_text,
                results_md=results_text_with_repo,
                discussion_md=discussion_text_with_repo,
                project_plan=project_plan,
                references_bib=references_bib,
                repo_url=repo_url,
            )

            if paper_tex and getattr(self.config, "enable_paper_polish", False):
                try:
                    paper_tex = polish_paper_tex(
                        paper_tex,
                        critique_model=self.config.editor_model,
                        edit_model=self.config.editor_model,
                    )
                except Exception:
                    pass

            paper_tex = sanitize_latex(paper_tex)

            return {
                "sections": {
                    "introduction": introduction_text,
                    "related_work": related_work_text,
                    "background": background_text,
                    "methods": methods_text,
                    "results": results_text_with_repo,
                    "discussion": discussion_text_with_repo,
                },
                "paper_tex": paper_tex,
                "references_bib": references_bib,
            }

        return await self.agent_run_sync(_write_sync)


class GithubAgent(Agent):
    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        super().__init__()
        self.config = config or AgentConfig()

    @action
    async def publish(
        self,
        question: str,
        source_dir: str,
        github_token: Optional[str],
        visibility: str,
        ignore_patterns: Optional[List[str]],
        branch: str = "main",
    ) -> Dict[str, Any]:
        def _publish_sync() -> Dict[str, Any]:
            repo_info: Optional[Dict[str, Any]] = None
            try:
                repo_info = create_repo(
                    question=question,
                    token=github_token,
                    visibility=visibility,
                )
            except Exception as exc:
                repo_info = {"created": False, "error": str(exc)}

            upload_info: Optional[Dict[str, Any]] = None
            if repo_info and repo_info.get("created") and repo_info.get("repo"):
                full_name = repo_info["repo"].get("full_name")
                if full_name:
                    upload_info = upload_run_artifacts(
                        repo_full_name=full_name,
                        source_dir=source_dir,
                        token=github_token,
                        branch=branch,
                        ignore_patterns=ignore_patterns,
                    )
                    repo_info["upload"] = upload_info
                    repo_info["repo"]["branch"] = branch
            return repo_info or {}

        return await self.agent_run_sync(_publish_sync)


# === Orchestrator ===


async def _run_orchestration(question: str, config: AgentConfig) -> Dict[str, Any]:
    # Create per-run directory structure upfront
    experiment_dirs = create_experiment_dirs(config.base_dir, question)

    async with await Manager.from_exchange_factory(
        factory=LocalExchangeFactory(),
        executors=ThreadPoolExecutor(),
    ) as manager:
        planning_handle = await manager.launch(PlanningAgent, kwargs={"config": config})
        codegen_handle = await manager.launch(CodegenAgent, kwargs={"config": config})
        validation_handle = await manager.launch(ValidationAgent, kwargs={"config": config})
        execution_handle = await manager.launch(ExecutionAgent, kwargs={"config": config})
        writing_handle = await manager.launch(WritingAgent, kwargs={"config": config})
        github_handle = await manager.launch(GithubAgent, kwargs={"config": config})

        project_plan = await planning_handle.plan(question)
        code_by_file = await codegen_handle.generate(question, project_plan)

        entrypoint_name: Optional[str] = None
        if isinstance(code_by_file, dict):
            entrypoint_name = project_plan.get("files", [{}])[0].get("name")
            for f in project_plan.get("files", []):
                if f.get("role") == "entrypoint":
                    entrypoint_name = f["name"]
                    break

        # Main attempt + retries loop
        total_attempts = 1 + max(getattr(config, "max_retries", 0), 0)
        attempts_meta: List[Dict[str, Any]] = []
        final_run_result: Dict[str, Any] = {
            "success": False,
            "stdout": "",
            "locals": {},
            "error": None,
        }

        for attempt_idx in range(total_attempts):
            combined_code_for_validation = _combine_project_code(project_plan, code_by_file)

            # Ensure environment.yaml is written for the runner
            env_code = code_by_file.get("environment.yaml")
            if env_code:
                env_path = Path(experiment_dirs["root_dir"]) / "environment.yaml"
                env_path.write_text(env_code, encoding="utf-8")

            validation_result = await validation_handle.validate(
                project_plan=project_plan,
                code_by_file=code_by_file,
                entrypoint_name=entrypoint_name,
            )
            all_valid = bool(validation_result.get("all_valid", False))
            validation_issues_by_file = validation_result.get("issues_by_file", {}) or {}

            if all_valid:
                final_run_result = await execution_handle.run(
                    combined_code=combined_code_for_validation,
                    work_dir=experiment_dirs["root_dir"],
                    entrypoint_name=entrypoint_name or "main.py",
                    env_tool=config.env_tool,
                    use_isolated_env=config.use_isolated_env,
                )
            else:
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

            if final_run_result.get("success") and all_valid:
                break

            if attempt_idx < total_attempts - 1 and entrypoint_name is not None:
                entry_code = code_by_file.get(entrypoint_name, "")
                issues_for_revision: List[str] = []
                if validation_issues_by_file:
                    for iss_list in validation_issues_by_file.values():
                        issues_for_revision.extend(iss_list)

                code_by_file[entrypoint_name] = await codegen_handle.revise(
                    question=question,
                    previous_code=entry_code,
                    run_result=final_run_result,
                    validation_issues=issues_for_revision,
                )
            else:
                break

        combined_code = _combine_project_code(project_plan, code_by_file)
        parsed_answer = await execution_handle.parse_answer(final_run_result.get("stdout", ""))
        explanation = await execution_handle.explain(question, combined_code, final_run_result)
        critic_report = await validation_handle.critic(
            question=question,
            parsed_answer=parsed_answer,
            explanation=explanation,
            run_result=final_run_result,
        )

        github_repo_info: Optional[Dict[str, Any]] = None
        repo_url_for_text: Optional[str] = None
        if getattr(config, "enable_github_publish", False) and final_run_result.get("success"):
            github_repo_info = await github_handle.publish(
                question=question,
                source_dir=experiment_dirs["root_dir"],
                github_token=getattr(config, "github_token", None),
                visibility=getattr(config, "github_visibility", "private"),
                ignore_patterns=getattr(config, "github_ignore_patterns", None),
                branch="main",
            )
            if github_repo_info and github_repo_info.get("repo"):
                repo_url_for_text = github_repo_info["repo"].get("html_url")

        writing_result = await writing_handle.write_all(
            question=question,
            project_plan=project_plan,
            code_by_file=code_by_file,
            parsed_answer=parsed_answer,
            run_result=final_run_result,
            critic_report=critic_report,
            explanation=explanation,
            repo_url=repo_url_for_text,
        )

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
            methods_text=writing_result["sections"]["methods"],
            results_text=writing_result["sections"]["results"],
            introduction_text=writing_result["sections"]["introduction"],
            background_text=writing_result["sections"]["background"],
            discussion_text=writing_result["sections"]["discussion"],
            paper_tex=writing_result["paper_tex"],
            references_bib=writing_result["references_bib"],
            attempts_meta=attempts_meta,
            related_work_text=writing_result["sections"]["related_work"],
            repo_info=github_repo_info,
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
            "sections": writing_result["sections"],
            "paper_tex": writing_result["paper_tex"],
            "references_bib": writing_result["references_bib"],
            "paths": paths,
            "repo_info": github_repo_info,
        }


def solve_with_academy_agents(question: str, config: Optional[AgentConfig] = None) -> Dict[str, Any]:
    """Synchronous entrypoint for running the Academy-orchestrated pipeline."""
    cfg = config or AgentConfig()
    return asyncio.run(_run_orchestration(question, cfg))


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Run the multi-agent pipeline via academy-py.")
    parser.add_argument("question", help="Problem statement for the agent to solve.")
    args = parser.parse_args()

    output = solve_with_academy_agents(args.question)
    print("QUESTION:")
    print(output["question"])
    for k, v in output["paths"].items():
        print(f"{k}: {v}")
