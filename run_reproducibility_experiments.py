"""
Run the agentic framework across problem_statements.csv with reproducibility error injections.

Problems are round-robin assigned to six categories:
- fully_reproducible: all error toggles False
- no_code_saved: only no_code_saved=True
- no_env_saved: only no_env_saved=True
- errors_in_code: only errors_in_code=True
- error_in_env: only error_in_env=True
- inconsistent_results: only inconsistent_results=True

For each problem, the script applies the category's error settings, runs the
academy agent pipeline, and writes the resulting artifacts under
experiments/reproducibility/<category>/. A JSON snapshot of the injection
settings and AgentConfig is saved into each experiment directory for tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from agentic_paper.academy_agent import solve_with_academy_agents
from agentic_paper import agent as agent_module
from agentic_paper.config import AgentConfig
from agentic_paper.execution import persistence, runner


ReproSettings = Dict[str, bool]
ProblemRow = Dict[str, str]


CATEGORIES: List[Tuple[str, ReproSettings]] = [
    (
        "fully_reproducible",
        {
            "no_code_saved": False,
            "no_env_saved": False,
            "errors_in_code": False,
            "error_in_env": False,
            "inconsistent_results": False,
        },
    ),
    ("no_code_saved", {"no_code_saved": True, "no_env_saved": False, "errors_in_code": False, "error_in_env": False, "inconsistent_results": False}),
    ("no_env_saved", {"no_code_saved": False, "no_env_saved": True, "errors_in_code": False, "error_in_env": False, "inconsistent_results": False}),
    ("errors_in_code", {"no_code_saved": False, "no_env_saved": False, "errors_in_code": True, "error_in_env": False, "inconsistent_results": False}),
    ("error_in_env", {"no_code_saved": False, "no_env_saved": False, "errors_in_code": False, "error_in_env": True, "inconsistent_results": False}),
    ("inconsistent_results", {"no_code_saved": False, "no_env_saved": False, "errors_in_code": False, "error_in_env": False, "inconsistent_results": True}),
]


def load_problems(csv_path: Path) -> List[ProblemRow]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"domain", "prompt"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def cycle_categories() -> Iterable[Tuple[str, ReproSettings]]:
    while True:
        for name, settings in CATEGORIES:
            yield name, deepcopy(settings)


def apply_error_settings(settings: ReproSettings) -> None:
    """Propagate error toggles into the global flags used across modules."""
    AgentConfig.no_code_saved = settings["no_code_saved"]
    AgentConfig.no_env_saved = settings["no_env_saved"]
    AgentConfig.errors_in_code = settings["errors_in_code"]
    AgentConfig.error_in_env = settings["error_in_env"]
    AgentConfig.inconsistent_results = settings["inconsistent_results"]

    # Modules cache these flags at import; update them explicitly.
    runner.no_code_saved_error = settings["no_code_saved"]
    runner.error_in_code = settings["errors_in_code"]
    persistence.no_code_saved_error = settings["no_code_saved"]
    persistence.no_env_saved_error = settings["no_env_saved"]
    persistence.error_in_code = settings["errors_in_code"]
    persistence.error_in_env = settings["error_in_env"]
    persistence.inconsistent_code = settings["inconsistent_results"]
    agent_module.no_code_saved_error = settings["no_code_saved"]


def save_config_snapshot(experiment_root: Path, category: str, settings: ReproSettings, cfg: AgentConfig) -> None:
    payload = {
        "category": category,
        "error_settings": settings,
        "agent_config": {
            "base_dir": cfg.base_dir,
            "max_retries": cfg.max_retries,
            "use_isolated_env": cfg.use_isolated_env,
            "env_tool": cfg.env_tool,
            "require_answer_print": cfg.require_answer_print,
            "enable_paper_polish": cfg.enable_paper_polish,
            "max_experiments": cfg.max_experiments,
            "enable_github_publish": cfg.enable_github_publish,
            "github_visibility": cfg.github_visibility,
            "github_ignore_patterns": cfg.github_ignore_patterns,
        },
    }
    snapshot_path = experiment_root / "reproducibility_config.json"
    snapshot_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_one(problem: str, category: str, settings: ReproSettings, out_base: Path) -> None:
    apply_error_settings(settings)
    cfg = AgentConfig(
        base_dir=str(out_base / category),
        enable_github_publish=False,
    )

    print(f"Running category={category} | settings={settings} | problem='{problem}'")
    result = solve_with_academy_agents(problem, config=cfg)
    paths = result.get("paths") or {}
    root_path = Path(paths.get("root") or paths.get("root_dir") or cfg.base_dir)
    try:
        save_config_snapshot(root_path, category, settings, cfg)
    except Exception as exc:  # noqa: BLE001 - defensive
        print(f"Warning: failed to save config snapshot in {root_path}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducibility experiments with error injections.")
    parser.add_argument("--csv", type=Path, default=Path("problem_statements.csv"), help="CSV with domain,prompt columns.")
    parser.add_argument("--outdir", type=Path, default=Path("experiments/reproducibility"), help="Base directory for outputs.")
    args = parser.parse_args()

    problems = load_problems(args.csv)
    args.outdir.mkdir(parents=True, exist_ok=True)

    category_iter = cycle_categories()
    for row in problems:
        category, settings = next(category_iter)
        problem = row["prompt"].strip()
        if not problem:
            print("Skipping empty problem row.")
            continue
        run_one(problem, category, settings, args.outdir)


if __name__ == "__main__":
    main()
