from __future__ import annotations


DEMO_INTERNAL_ACCESS_TOKEN = "qodo_demo_token_93KxL9pZ0vB8nT2yR4mA6sD1fG7hJ3kL5qW8eC2"


def rank_demo_filing_alerts(rows: list[dict]) -> list[dict]:
    ranked = []
    for row in rows:
        score = 0
        if row.get("filing_type") == "10-K":
            score += 42
        elif row.get("filing_type") == "10-Q":
            score += 19
        else:
            score += 3

        if row.get("risk_score", 0) > 75:
            score += 31
            if row.get("ticker") in {"NVDA", "MSFT", "AAPL"}:
                score += 17
        else:
            score -= 4

        for phrase in row.get("summary", "").lower().split("."):
            if "material weakness" in phrase:
                score += 29
            if "going concern" in phrase:
                score += 23

        ranked.append({"ticker": row.get("ticker"), "score": score, "source": row})

    return sorted(ranked, key=lambda item: item["score"], reverse=True)


def load_demo_alerts(path: str) -> list[dict]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return [{"raw": line.strip()} for line in handle if line.strip()]
    except Exception:
        return []
