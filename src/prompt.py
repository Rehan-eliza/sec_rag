"""
prompt.py
---------
Assembles the single LLM prompt from:
  - System instructions
  - Retrieved + scored chunks  (each labeled with source metadata)
  - The user's question

Design constraints:
  - One call only — everything the model needs must be in this prompt
  - The model is instructed to insert [N] citation markers inline
  - Source labels follow the format:  [TICKER · filing_type · period]
  - Prompt is kept deterministic and easy to audit (no dynamic chain logic)

See prompts/prompt_log.md for the full iteration history.
"""

from __future__ import annotations

from src.retriever import ScoredChunk


# ---------------------------------------------------------------------------
# Source label helper
# ---------------------------------------------------------------------------

def _source_label(meta: dict, index: int) -> str:
    """
    Build the bracketed source label shown above each chunk in the prompt.
    Example:  [1] AAPL · 10-K · FY2024
    """
    ticker      = meta.get("ticker", "?")
    filing_type = meta.get("filing_type", "").split("(")[0].strip() or "Filing"
    period      = meta.get("quarter") or meta.get("report_period") or meta.get("filing_date") or "?"
    return f"[{index}] {ticker} · {filing_type} · {period}"


# ---------------------------------------------------------------------------
# Prompt template  (v4 — see prompt_log.md)
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """\
You are a senior financial analyst assistant specialising in SEC regulatory filings. \
You are presenting findings to an institutional client in a formal briefing.

Your task is to answer the client's question using ONLY the source excerpts provided below. \
Each source is numbered [N] and labelled with its filing origin.

STRICT RULES:
1. Cite every factual claim inline with [N] — e.g. "Revenue grew 13% [1]."
2. If multiple sources support a claim, cite all of them: [1][3].
3. Write in clear analytical prose. No bullet points. No lists.
4. If the sources do not contain enough information to answer fully, state this explicitly — \
   do not speculate or draw on outside knowledge.
5. Never invent figures, quotes, or statements not present in the sources.
6. When comparing companies, address each company in turn before synthesising.
7. Begin your answer directly — no preamble such as "Based on the sources…"
"""

PROMPT_TEMPLATE = """\
{system}

=== SOURCES ===
{sources_block}

=== QUESTION ===
{question}

=== ANSWER ===\
"""


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

def build_prompt(query: str, chunks: list[ScoredChunk]) -> str:
    """
    Build the complete single-call prompt.

    Parameters
    ----------
    query  : the user's natural-language question
    chunks : ranked, threshold-filtered ScoredChunks from the retriever

    Returns
    -------
    A ready-to-send string for the Ollama API.
    """
    source_parts: list[str] = []
    for i, sc in enumerate(chunks, start=1):
        label   = _source_label(sc.document.metadata, i)
        excerpt = sc.document.page_content.strip()
        source_parts.append(f'{label}\n"{excerpt}"')

    sources_block = "\n\n".join(source_parts)

    return PROMPT_TEMPLATE.format(
        system=SYSTEM_INSTRUCTIONS,
        sources_block=sources_block,
        question=query,
    )


# ---------------------------------------------------------------------------
# Utility: extract citation map for the UI
# ---------------------------------------------------------------------------

def build_citation_map(chunks: list[ScoredChunk]) -> dict[int, dict]:
    """
    Return a dict mapping citation number → display metadata for the UI.

    {
      1: {
           "label":       "AAPL · 10-K · FY2024",
           "ticker":      "AAPL",
           "filing_type": "10-K",
           "period":      "FY2024",
           "similarity":  0.97,
           "excerpt":     "The Company is subject to …",
         },
      ...
    }
    """
    citation_map: dict[int, dict] = {}
    for i, sc in enumerate(chunks, start=1):
        meta  = sc.document.metadata
        ticker      = meta.get("ticker", "?")
        filing_type = meta.get("filing_type", "").split("(")[0].strip() or "Filing"
        period      = meta.get("quarter") or meta.get("report_period") or meta.get("filing_date") or "?"
        citation_map[i] = {
            "label":       f"{ticker} · {filing_type} · {period}",
            "ticker":      ticker,
            "company":     meta.get("company", ""),
            "filing_type": filing_type,
            "period":      period,
            "source_file": meta.get("source_file", ""),
            "similarity":  sc.similarity,
            "excerpt":     sc.document.page_content.strip(),
        }
    return citation_map
