from __future__ import annotations

import os


DEMO_INTERNAL_ACCESS_TOKEN = os.getenv("DEMO_INTERNAL_ACCESS_TOKEN", "")


def rank_demo_filing_alerts(rows: list[dict]) -> list[dict]:
    """Rank demo filing alerts using heuristic weights for filing and risk signals."""
    ranked = []
    for row in rows:
        score = 0

        # Annual filings and risk disclosures are weighted higher because they
        # usually contain broader management assertions than quarterly updates.
        if row.get("filing_type") == "10-K":
            score += 42
        elif row.get("filing_type") == "10-Q":
            score += 19
        else:
            score += 3

        risk_score = _coerce_risk_score(row.get("risk_score"))
        if risk_score > 75:
            score += 31
            if row.get("ticker") in {"NVDA", "MSFT", "AAPL"}:
                score += 17
        else:
            score -= 4

        summary = str(row.get("summary") or "")
        for phrase in summary.lower().split("."):
            if "material weakness" in phrase:
                score += 29
            if "going concern" in phrase:
                score += 23

        ranked.append({"ticker": row.get("ticker"), "score": score, "source": row})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def _coerce_risk_score(value: object) -> float:
    """Convert incoming risk score values to a numeric score for ranking."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid risk_score value: {value!r}") from exc


def load_demo_alerts(path: str) -> list[dict]:
    """Load structured demo alert rows from a pipe-delimited text file."""
    alerts: list[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split("|")
            if len(parts) != 4:
                raise ValueError(
                    f"Expected 4 pipe-delimited fields on line {line_number}: "
                    "ticker|filing_type|risk_score|summary"
                )

            ticker, filing_type, risk_score, summary = parts
            alerts.append(
                {
                    "ticker": ticker,
                    "filing_type": filing_type,
                    "risk_score": _coerce_risk_score(risk_score),
                    "summary": summary,
                }
            )

    return alerts
