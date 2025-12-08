"""
Generate research-paper-style .txt files for each problem statement in a CSV.

For every row in problem_statements.csv, the script loads llm_prompt.txt,
replaces the {PROBLEM} placeholder, and sends the resulting prompt to the
inference LLM endpoint. The returned paper text is saved to an output
directory as a .txt file.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

from agentic_paper.utils.llm_client import call_llm, DEFAULT_MODEL


def load_template(template_path: Path) -> str:
    template = template_path.read_text(encoding="utf-8")
    if "{PROBLEM}" not in template:
        raise ValueError("Template is missing the {PROBLEM} placeholder.")
    return template


def load_problems(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required_fields = {"domain", "prompt"}
        missing = required_fields - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
        return list(reader)


def slugify(text: str, max_length: int = 60) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip("_")
    return slug or "paper"


def build_messages(prompt_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are a meticulous technical researcher who writes full research papers "
                "with standard sections and detailed analysis."
            ),
        },
        {"role": "user", "content": prompt_text},
    ]


def generate_paper(prompt_text: str, model: str, temperature: float, max_tokens: int | None) -> str:
    messages = build_messages(prompt_text)
    attempts = 3
    for attempt in range(1, attempts + 1):
        paper = call_llm(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if paper and len(paper.strip()) >= 50:
            return paper
        print(f"Attempt {attempt} returned empty/short output; retrying...")
    raise RuntimeError("LLM returned empty or too-short paper after 3 attempts.")


def write_paper(output_dir: Path, domain: str, prompt: str, index: int, paper_text: str) -> Path:
    slug = slugify(prompt)
    filename = f"{index:02d}_{domain}_{slug}.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    output_path.write_text(paper_text, encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate papers from problem_statements.csv using llm_prompt.txt.")
    parser.add_argument("--csv", type=Path, default=Path("problem_statements.csv"), help="Path to CSV of problem statements.")
    parser.add_argument("--prompt", type=Path, default=Path("llm_prompt.txt"), help="Path to template prompt containing {PROBLEM}.")
    parser.add_argument("--outdir", type=Path, default=Path("generated_papers"), help="Directory to write paper .txt files.")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model name for the inference endpoint.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens for the LLM response.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip prompts whose output file already exists.")
    args = parser.parse_args()

    template = load_template(args.prompt)
    problems = load_problems(args.csv)

    for idx, row in enumerate(problems, start=1):
        domain = row["domain"].strip().lower() or "general"
        problem_statement = row["prompt"].strip()
        if not problem_statement:
            print(f"[{idx}] Skipping empty prompt row.")
            continue

        if "{PROBLEM}" not in template:
            raise ValueError("Template is missing the {PROBLEM} placeholder.")

        formatted_prompt = template.replace("{PROBLEM}", problem_statement)
        output_path = args.outdir / f"{idx:02d}_{domain}_{slugify(problem_statement)}.txt"
        if args.skip_existing and output_path.exists():
            print(f"[{idx}] Skipping existing {output_path}")
            continue

        print(f"[{idx}] Generating paper for: {problem_statement}")
        try:
            paper = generate_paper(
                prompt_text=formatted_prompt,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            saved_to = write_paper(args.outdir, domain, problem_statement, idx, paper)
            print(f"[{idx}] Saved to {saved_to}")
        except Exception as exc:  # noqa: BLE001 - want any error surfaced
            print(f"[{idx}] Failed to generate paper: {exc}")


if __name__ == "__main__":
    main()
