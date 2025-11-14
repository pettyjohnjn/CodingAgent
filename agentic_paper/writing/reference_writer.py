from typing import Dict, Any
from ..utils.llm_client import call_llm


def generate_references_bib(
    question: str,
    introduction_text: str,
    methods_text: str,
    results_text: str,
    discussion_text: str,
    paper_meta: Dict[str, Any] | None = None,
) -> str:
    """
    Produce a BibTeX file content (as a string) containing several references
    relevant to the topic of the paper.

    The output is intended to be written to `references.bib` and used with:

        \\bibliographystyle{{ACM-Reference-Format}}
        \\bibliography{{references}}
    """
    pm = paper_meta or {}
    domain_context = pm.get("domain_context", "")
    research_question = pm.get("research_question", question)

    system_msg = {
        "role": "system",
        "content": (
            "You are generating a BibTeX bibliography for an ACM-style scientific paper. "
            "You will be given the research question, the scientific domain, and draft sections "
            "of the paper (Introduction, Methods, Results, Discussion).\n\n"
            "Requirements:\n"
            "- Produce 3–8 plausible references in valid BibTeX format.\n"
            "- Use standard BibTeX entry types such as @article, @book, or @inproceedings.\n"
            "- Use concise citation keys (e.g., 'press2007numerical', 'stoer1993introduction').\n"
            "- Choose references that are plausible for the described domain and topic.\n"
            "- Output ONLY the BibTeX entries, with no explanations, comments, or surrounding text.\n"
        ),
    }

    user_content = (
        f"High-level user question:\n{question}\n\n"
        f"Framed research question:\n{research_question}\n\n"
        f"Domain context (e.g., numerical analysis, epidemiology, physics):\n{domain_context}\n\n"
        "Draft Introduction:\n"
        "----- INTRO START -----\n"
        f"{introduction_text}\n"
        "----- INTRO END -------\n\n"
        "Draft Methods:\n"
        "----- METHODS START ----\n"
        f"{methods_text}\n"
        "----- METHODS END ------\n\n"
        "Draft Results:\n"
        "----- RESULTS START ----\n"
        f"{results_text}\n"
        "----- RESULTS END ------\n\n"
        "Draft Discussion:\n"
        "----- DISCUSSION START -\n"
        f"{discussion_text}\n"
        "----- DISCUSSION END ---\n\n"
        "Now output only the BibTeX entries for a references.bib file."
    )

    user_msg = {"role": "user", "content": user_content}
    bib_text = call_llm([system_msg, user_msg])
    return bib_text.strip()