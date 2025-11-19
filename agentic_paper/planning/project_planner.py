from __future__ import annotations

from typing import Dict, Any, List

from .experiment_planner import plan_experiments
from ..config import AgentConfig


def _normalize_metric_names(experiment_plan: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure that metric names are globally unique across experiments.

    If the same metric name appears in multiple experiments, suffix it with
    the experiment id (or index) so the validator can distinguish them.

    Example:
      - exp1 has metric "approx_value_N320"
      - exp2 also has metric "approx_value_N320"

    After normalization:
      - exp1: "approx_value_N320_exp1"
      - exp2: "approx_value_N320_exp2"
    """
    experiments = experiment_plan.get("experiments", [])
    seen: set[str] = set()

    normalized_experiments: List[Dict[str, Any]] = []

    for idx, exp in enumerate(experiments):
        exp_id = exp.get("id") or f"exp{idx + 1}"
        safe_suffix = exp_id.replace(" ", "_")

        outputs = exp.get("outputs") or {}
        metrics = outputs.get("metrics") or []
        new_metrics: List[Dict[str, Any]] = []

        for m in metrics:
            name = m.get("name")
            if not name:
                new_metrics.append(m)
                continue

            new_name = name
            if new_name in seen:
                # Make it unique by suffixing with experiment id
                new_name = f"{name}_{safe_suffix}"

            seen.add(new_name)
            new_m = dict(m)
            new_m["name"] = new_name
            new_metrics.append(new_m)

        outputs["metrics"] = new_metrics
        exp["outputs"] = outputs
        normalized_experiments.append(exp)

    experiment_plan["experiments"] = normalized_experiments
    return experiment_plan


def plan_project(question: str, config: AgentConfig) -> Dict[str, Any]:
    """
    Return a project plan for the given question, plus a high-level
    experiment specification.

    NEW DESIGN:
    - Single Python file (entrypoint only), no enforced helpers.py.
    - The entrypoint file is responsible for:
        * Implementing any needed helper functions locally,
        * Running all experiments described in project_plan['experiments'],
        * Generating all requested plots with the exact filenames,
        * Computing and printing all requested metrics,
        * Printing the final numeric answer with: print('Answer:', value).
    """

    # Ask the LLM-based experiment planner to propose experiments
    raw_experiment_plan = plan_experiments(
        question,
        model=config.planner_model,
        max_experiments=config.max_experiments,
    )

    # Normalize metric names so they are globally unique and validator-friendly
    experiment_plan = _normalize_metric_names(raw_experiment_plan)

    # Single-file layout: one entrypoint script
    files: List[Dict[str, Any]] = [
        {
            "name": "main.py",
            "role": "entrypoint",
            "description": (
                "Single-file solution for the entire project. "
                "Implement any necessary helper functions and numerical routines "
                "directly in this file. Running this script must:\n"
                "- Execute all experiments described in project_plan['experiments'].\n"
                "- Generate all requested plots using matplotlib, saving them with the "
                "exact filenames specified in the experiment plan.\n"
                "- Compute and (optionally) print any requested metrics.\n"
                "- Finally, print the primary numeric answer on a line of the form:\n"
                "    print('Answer:', value)\n"
                "No additional modules or files are required; keep everything in this "
                "one script."
            ),
        },
        {
            "name": "environment.yaml",
            "role": "env",
            "description": (
                "Conda environment specification required to run main.py. "
                "Use a valid YAML conda env format with a name, channels, and "
                "dependencies. Include a pinned Python version and only the libraries "
                "that are actually imported in main.py (e.g., numpy, matplotlib, etc.)."
            ),
        },
    ]

    project_plan: Dict[str, Any] = {
        "mode": "single_file_experiment",
        "question": question,
        "files": files,
        # Attach *normalized* experiments
        "experiments": experiment_plan.get("experiments", []),
    }

    return project_plan