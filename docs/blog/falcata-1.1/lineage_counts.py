"""Recompute the lineage counts quoted in benchmarks.md with the explorer's own rules.

Mirrors lineage-explorer.js (assessment, speed class and replay source grouping). Run from this directory or with
the archive path as the only argument:
    python lineage_counts.py [lineage-data.json]
"""

import collections
import json
import sys
from pathlib import Path


def is_number(value):
    """Return True for a JSON number (not a boolean)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def assessment(node):
    """Return (passed | blocked | unknown, reasons) as the explorer's assessment() does."""
    checks = node.get("checks") or {}
    reasons = []
    if checks.get("later_expanded_coverage_passed") is False:
        reasons.append("coverage")
    if checks.get("review") == "reject" or node.get("outcome") == "rejected":
        reasons.append("rejected")
    if checks.get("byte_exact_valid") is False:
        reasons.append("byte")
    if checks.get("md5_ok") is False:
        reasons.append("md5")
    if checks.get("lattice_ok") is not None and checks.get("lattice_ok") != "ok":
        reasons.append("lattice")
    if checks.get("suspect") is True:
        reasons.append("suspect")
    if node.get("outcome") == "failed" and not reasons:
        reasons.append("failed")
    if reasons:
        return "blocked", reasons
    passed = (node.get("status") == "valid" and checks.get("byte_exact_valid") is True) or (
        node.get("status") in ("ok", "regressed") and checks.get("md5_ok") is True and checks.get("lattice_ok") == "ok"
    )
    return ("passed" if passed else "unknown"), []


def counts(data):
    """Return the coverage, grouping, speed and check counts of one export."""
    cand = [n for n in data["nodes"] if n["kind"] == "candidate"]
    replay = [n for n in cand if n.get("source_hash")]

    def group(node):
        return (node["campaign"], node["source_hash"]) if node.get("source_hash") else node["id"]

    kinds = {n["id"]: n["kind"] for n in data["nodes"]}
    integ = [e for e in data["edges"] if e["kind"] == "integration"]
    above = [n for n in cand if is_number(n.get("ratio")) and n["ratio"] > 1]
    not_improved = [n for n in above if n.get("outcome") != "improved"]
    cov = data["coverage"]
    return {
        "campaigns": len(data["campaigns"]),
        "records": len(cand),
        "archived_records": cov["archived_records"],
        "library_records": cov["agentic_archive_records"],
        "replay_records": len(replay),
        "started_unarchived": cov["started_unarchived"],
        "baselines": cov["baselines"],
        "milestones": len(data["milestones"]),
        "references": cov["reference_nodes"],
        "nodes": len(data["nodes"]),
        "edges": len(data["edges"]),
        "edge_kinds": dict(collections.Counter(e["kind"] for e in data["edges"])),
        "milestone_to_baseline_edges": sum(kinds[e["source"]] == "milestone" for e in integ),
        "candidate_to_milestone_edges": sum(kinds[e["target"]] == "milestone" for e in integ),
        "excluded_archive_records": cov["excluded_archive_records"],
        "replay_source_groups": len({group(n) for n in replay}),
        "grouped_entries": len({group(n) for n in cand}),
        "above_baseline": len(above),
        "above_baseline_not_improved": len(not_improved),
        "above_baseline_not_improved_by_outcome": dict(collections.Counter(n.get("outcome") for n in not_improved)),
        "above_baseline_checks_passed": sum(assessment(n)[0] == "passed" for n in above),
        "above_baseline_review_rejected": sum("rejected" in assessment(n)[1] for n in above),
        "above_baseline_later_coverage_failed": sum("coverage" in assessment(n)[1] for n in above),
        "checks": dict(collections.Counter(assessment(n)[0] for n in cand)),
        "public_commits": len(data["public_commits"]),
    }


def main():
    """Print the counts as JSON."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "lineage-data.json"
    sys.stdout.write(json.dumps(counts(json.loads(path.read_text())), indent=1) + "\n")


if __name__ == "__main__":
    main()
