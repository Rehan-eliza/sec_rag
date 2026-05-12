# Prompt Iteration Log

Documents the evolution of the system prompt and prompt template used for
the single Ollama LLM call. Each entry records what changed, why, and what
the observed effect was.

---

## v1 — Baseline

**Date:** initial build  
**File:** `src/prompt.py`

**What it was:**
Simple instruction to answer from provided sources with no citation rules.

```
You are a financial analyst. Answer the question using the sources below.
Sources: {sources}
Question: {question}
```

**Problems observed:**
- Model answered from general knowledge when sources were thin.
- No citation markers — impossible to trace claims to filings.
- Answer structure varied wildly across question types.

---

## v2 — Add citation instruction

**Change:** Instructed model to insert `[N]` markers after every claim.  
**Why:** Traceability is a core reliability requirement for the client demo.

**Problems observed:**
- Model sometimes grouped all citations at the end of a paragraph rather than inline.
- Occasionally cited `[1]` for everything regardless of which source supported it.

---

## v3 — Add role + strict rules block

**Change:** Added senior analyst persona. Added numbered STRICT RULES section
explicitly prohibiting speculation and requiring per-claim inline citation.  
**Why:** Role framing improved output register. Numbered rules reduced citation
grouping behaviour.

**Problems observed:**
- Model still opened answers with "Based on the sources provided…" boilerplate.
- Bullet-point answers appeared for comparison questions.

---

## v4 — Final (current)

**Change:**
- Added rule 7: "Begin your answer directly — no preamble."
- Added rule 3: "Write in clear analytical prose. No bullet points. No lists."
- Added rule 6: "When comparing companies, address each company in turn before synthesising."
- Source block format changed to labeled `[N] TICKER · type · period` headers
  above each excerpt so the model can see provenance before reading content.

**Why:**  
The preamble rule eliminates filler that wastes context. The prose rule
produces more readable output for a client meeting. The comparison rule
gives structure to multi-company questions without allowing bullet lists.
The labeled source header means the model doesn't have to infer provenance
from content — it's explicit.

**Current status:** In production in `src/prompt.py`.

---

## Candidate v5 — ideas to test

- Add instruction: "If two sources contradict each other, note the discrepancy explicitly."
- Add instruction: "Where a figure is cited (revenue, margin, etc.) always include the unit."
- Test shorter `num_predict` to force more concise answers and measure quality trade-off.
- Experiment with placing the question *before* the sources to prime the model's attention.
