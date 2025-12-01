"""
Assembler for an ACM-style LaTeX paper.

- Takes Markdown sections (Introduction, Methods, Results, Discussion, optional Related Work).
- Uses the project_plan's experiments to insert LaTeX figures that reference
  any PNGs (or other image files) created by the experiments.
- Uses references_bib (BibTeX) to insert at least one \cite{...} into the body.
"""

from __future__ import annotations

import re
from typing import Dict, Any, List

from ..utils.llm_client import call_llm
from ..utils.latex_sanitizer import sanitize_for_latex, markdown_to_latex_body
from ..config import AgentConfig


def strip_markdown_heading(md: str, expected_prefix: str) -> str:
    """
    Remove the first line if it matches an expected Markdown heading prefix
    (e.g., '## Introduction').
    """
    if not md:
        return ""
    lines = md.splitlines()
    if not lines:
        return ""
    first = lines[0].lstrip()
    if first.lower().startswith(expected_prefix.lower()):
        return "\n".join(lines[1:]).lstrip("\n")
    return md


def _generate_title_and_abstract(
    question: str,
    introduction_md: str,
    methods_md: str,
    results_md: str,
) -> Dict[str, str]:
    """
    Ask the LLM for a short title and abstract given the main sections.

    Returns:
        {
            "title": "...",
            "abstract": "..."
        }
    """
    system_msg = {
        "role": "system",
        "content": (
            "You are helping to prepare an ACM-style scientific paper. "
            "Given the research question and draft Introduction, Methods, and Results "
            "sections, propose a concise, informative title and a short abstract.\n\n"
            "Requirements:\n"
            "- Use ONLY ASCII characters; do NOT use Unicode symbols such as pi, integral "
            "signs, superscript/subscript characters, or smart quotes.\n"
            "- Use LaTeX math syntax when needed (e.g., $\\int_0^\\pi \\sin x\\,dx$).\n"
            "- The title should be on a single line, no more than about 15 words.\n"
            "- The abstract should be 3–6 sentences suitable for an ACM conference paper.\n"
            "- Do NOT include LaTeX environments or commands other than math symbols "
            "inside the abstract; it should be plain text with optional inline math.\n"
            "- Return plain text only in the format:\n"
            "  TITLE: <title here>\n"
            "  ABSTRACT: <abstract here>\n"
        ),
    }

    user_content = (
        f"Research question:\n{question}\n\n"
        "Draft Introduction:\n"
        "----- INTRO START -----\n"
        f"{introduction_md}\n"
        "----- INTRO END -------\n\n"
        "Draft Methods:\n"
        "----- METHODS START ----\n"
        f"{methods_md}\n"
        "----- METHODS END ------\n\n"
        "Draft Results:\n"
        "----- RESULTS START ----\n"
        f"{results_md}\n"
        "----- RESULTS END ------\n\n"
        "Respond in the format:\n"
        "TITLE: <title here>\n"
        "ABSTRACT: <abstract text here>\n"
    )

    user_msg = {"role": "user", "content": user_content}
    content = call_llm([system_msg, user_msg]).strip()

    title = "Automated Numerical Experiment"
    abstract = "This paper presents an automatically generated numerical experiment."

    # Simple parsing
    m_title = re.search(r"TITLE:\s*(.+)", content)
    m_abs = re.search(r"ABSTRACT:\s*(.+)", content, flags=re.DOTALL)
    if m_title:
        title = m_title.group(1).strip()
    if m_abs:
        abstract = m_abs.group(1).strip()

    # Final safety pass: sanitize for LaTeX (ASCII + safe macros)
    title = sanitize_for_latex(title)
    abstract = sanitize_for_latex(abstract)

    return {"title": title, "abstract": abstract}


def _build_figures_latex(project_plan: Dict[str, Any]) -> str:
    """
    Build LaTeX figure environments for any plots specified in project_plan['experiments'].

    Assumes the image files (e.g., PNGs) live in the same directory as paper.tex.
    """
    experiments = project_plan.get("experiments", []) if project_plan else []
    lines: List[str] = []

    for exp in experiments:
        exp_name = sanitize_for_latex(exp.get("name", "") or "")
        outputs = exp.get("outputs", {})
        plots = outputs.get("plots", [])

        for plot in plots:
            fn = plot.get("filename", "")
            if not fn:
                continue
            fn = sanitize_for_latex(fn).strip()
            if not fn:
                continue

            desc = plot.get("description", "") or exp_name or "Experiment figure"
            caption = sanitize_for_latex(desc)

            lines.append("\\begin{figure}[t]")
            lines.append("\\centering")
            lines.append(f"\\includegraphics[width=\\linewidth]{{{fn}}}")
            lines.append(f"\\caption{{{caption}}}")
            if exp_name:
                label = exp_name.replace(" ", "-").lower()
                label = sanitize_for_latex(label)
                lines.append(f"\\label{{fig:{label}}}")
            lines.append("\\end{figure}")
            lines.append("")  # blank line between figures

    return "\n".join(lines)


def _extract_bib_keys(references_bib: str) -> List[str]:
    """
    Extract BibTeX keys from a references.bib string.
    """
    if not references_bib:
        return []
    pattern = re.compile(r"@\w+\{([^,]+),")
    keys = pattern.findall(references_bib)
    # Deduplicate while preserving order
    seen = set()
    unique: List[str] = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique.append(k)
    return unique


def _make_citation_clause(references_bib: str) -> str:
    """
    Build a short LaTeX citation clause like '~\\cite{key1,key2}' using keys
    extracted from references_bib. Returns an empty string if no keys found.
    """
    keys = _extract_bib_keys(references_bib)
    if not keys:
        return ""
    subset = keys[:4]
    cite_body = ",".join(subset)
    return f"~\\cite{{{cite_body}}}"


def assemble_acm_paper_tex(
    question: str,
    intro_md: str,
    background_md: str,
    related_work_md: str,
    methods_md: str,
    results_md: str,
    discussion_md: str,
    project_plan: Dict[str, Any] | None = None,
    references_bib: str | None = None,
    repo_url: str | None = None,
) -> str:
    """
    Assemble a full ACM-style LaTeX source using acmart, sigconf style.

    We assume the caller has already generated Markdown sections:
        - intro_md: starts with '## Introduction'
        - methods_md: starts with '## Methods'
        - results_md: starts with '## Results'
        - discussion_md: starts with '## Discussion and Conclusion'
        - related_work_md (optional): starts with '## Background and Related Work'
    """
    # Strip top-level Markdown headings from sections
    intro_body_md = strip_markdown_heading(intro_md, "## Introduction")
    background_body_md = strip_markdown_heading(background_md, "## Background")
    related_body_md = strip_markdown_heading(related_work_md, "## Related Work")
    methods_body_md = strip_markdown_heading(methods_md, "## Methods")
    results_body_md = strip_markdown_heading(results_md, "## Results")
    discussion_body_md = strip_markdown_heading(
        discussion_md, "## Discussion and Conclusion"
    )

    # Convert Markdown-ish content to LaTeX-like body
    intro_body_tex = markdown_to_latex_body(intro_body_md)
    background_body_tex = markdown_to_latex_body(background_body_md)
    related_body_tex = markdown_to_latex_body(related_body_md)
    methods_body_tex = markdown_to_latex_body(methods_body_md)
    results_body_tex = markdown_to_latex_body(results_body_md)
    discussion_body_tex = markdown_to_latex_body(discussion_body_md)

    # Build a citation clause from the BibTeX keys, if available
    citation_clause = _make_citation_clause(references_bib or "")

    # Append a short literature sentence with an actual \cite{...} to Introduction
    if citation_clause:
        intro_body_tex = (
            intro_body_tex
            + "\n\nThis work is consistent with standard treatments of numerical "
            f"integration and scientific computing{citation_clause}.\n"
        )

    # Get title and abstract from LLM (then sanitized)
    meta = _generate_title_and_abstract(
        question=question,
        introduction_md=intro_md,
        methods_md=methods_md,
        results_md=results_md,
    )
    title = meta["title"]
    abstract = meta["abstract"]

    # Build figure environments from experiment plots
    figures_tex = _build_figures_latex(project_plan or {})

    availability_tex = ""
    if repo_url and not AgentConfig.no_code_saved:
        safe_repo_url = sanitize_for_latex(repo_url)
        availability_tex = (
            "\n\\section{Code Availability}\n"
            f"The full generated code, experiment scripts, and paper sources are available at \\url{{{safe_repo_url}}}.\n"
        )

    # Optional related-work section
    related_section_tex = ""
    if related_body_tex:
        related_section_tex = (
            "\n\\section{Background and Related Work}\n"
            f"{related_body_tex}\n"
        )

    tex = f"""% Auto-generated ACM-style LaTeX document
\\documentclass[sigconf]{{acmart}}

% Metadata
\\title{{{title}}}

\\author{{Automated Research Agent}}
\\affiliation{{%
  \\institution{{Automated System}}
  \\country{{}}
}}
\\email{{auto-generated@example.com}}

\\begin{{document}}

\\begin{{abstract}}
{abstract}
\\end{{abstract}}

\\maketitle

\\section{{Introduction}}
{intro_body_tex}

\\section{{Background}}
{background_body_tex}

\\section{{Related Work}}
{related_body_tex}

\\section{{Methods}}
{methods_body_tex}

\\section{{Results}}
{results_body_tex}

{figures_tex}

\\section{{Discussion and Conclusion}}
{discussion_body_tex}

{availability_tex}

\\bibliographystyle{{ACM-Reference-Format}}
\\bibliography{{references}}

\\end{{document}}
"""
    # Final safety pass over the entire document (ensures no stray Unicode)
    tex = sanitize_for_latex(tex)
    return tex
