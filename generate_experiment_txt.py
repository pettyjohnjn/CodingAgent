"""
Create a single consolidated .txt file for each experiment run in experiments/.

Only processes top-level experiment directories whose names start with
the timestamp prefix "20251205". For each run, it concatenates the saved
markdown sections (if present) in a logical order and writes a
<experiment_dir_name>.txt file into that same experiment directory.
"""

from __future__ import annotations

from pathlib import Path


SECTION_FILES = [
    ("Title", None),  # filled from state.json question if available
    ("Introduction", "introduction.md"),
    ("Background", "background.md"),
    ("Related Work", "related_work.md"),
    ("Methods", "methods.md"),
    ("Results", "results.md"),
    ("Discussion", "discussion.md"),
    ("Explanation", "explanation.md"),
]


def load_state_question(exp_dir: Path) -> str | None:
    state_path = exp_dir / "state.json"
    if not state_path.exists():
        return None
    try:
        import json  # local import to keep module lightweight
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data.get("question")
    except Exception:
        return None


def build_txt(exp_dir: Path) -> str:
    parts: list[str] = []
    question = load_state_question(exp_dir)

    for header, filename in SECTION_FILES:
        if header == "Title":
            if question:
                parts.append(question.strip())
                parts.append("")  # blank line after title
            continue

        if filename is None:
            continue
        md_path = exp_dir / "markdown" / filename
        if not md_path.exists():
            continue
        content = md_path.read_text(encoding="utf-8").strip()
        if not content:
            continue
        parts.append(header)
        parts.append(content)
        parts.append("")  # blank line between sections

    return "\n".join(parts).strip() + "\n"


def main() -> None:
    experiments_root = Path("experiments")
    if not experiments_root.exists():
        raise SystemExit("experiments/ directory not found.")

    for exp_dir in experiments_root.iterdir():
        if not exp_dir.is_dir():
            continue
        if not exp_dir.name.startswith("20251205"):
            continue
        txt_path = exp_dir / f"{exp_dir.name}.txt"
        txt_body = build_txt(exp_dir)
        txt_path.write_text(txt_body, encoding="utf-8")
        print(f"Wrote {txt_path}")


if __name__ == "__main__":
    main()
