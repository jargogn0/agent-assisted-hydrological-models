#!/usr/bin/env python3
"""Lineage-correct trajectory extraction: group iterations by their anchoring
baseline using the TAG timestamp (log ts is written after training completes)."""
import json, re
from pathlib import Path

AR = Path(__file__).resolve().parent / "paper4_pipeline/outputs/autoresearch"


def tagtime(tag: str) -> str:
    m = re.search(r"(\d{8}_\d{6})", tag or "")
    return m.group(1) if m else "0"


def campaigns(scope: str) -> list[dict]:
    """Return every anchored campaign for a scope, oldest first."""
    recs = [json.loads(l) for l in (AR / "agent_log.jsonl").read_text().splitlines() if l.strip()]
    rs = [r for r in recs if r.get("scope") == scope]
    bases = [r for r in rs if r.get("outcome") == "baseline" and r.get("composite_score") is not None]
    out = []
    for i, base in enumerate(bases):
        t0 = tagtime(base["tag"])
        t1 = tagtime(bases[i + 1]["tag"]) if i + 1 < len(bases) else "99999999_999999"
        it = [r for r in rs if r.get("outcome") != "baseline" and t0 <= tagtime(r["tag"]) < t1]
        out.append({"baseline": base, "iterations": it,
                    "promoted": [r for r in it if r.get("outcome") == "promote"]})
    return out
