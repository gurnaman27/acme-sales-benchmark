"""Failure Taxonomy Analysis & Trajectory Case Studies.

Analyzes failed trajectories across participant agents, classifies root causes
using the Phase 3 taxonomy, and generates structured failure reports and case studies.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

from acme.evaluation.failure_taxonomy import FailureType
from acme.tasks.task_library import get_task_by_id


def analyze_failures(
    results_json_path: str = "results/results.json",
    out_dir: str = "results",
    agent_name: str | None = None,
    output_prefix: str | None = None,
) -> dict[str, Any]:
    """Analyze failures from a benchmark results JSON file.

    Supports both single-agent run output format and multi-agent comparison JSON format.
    """
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(results_json_path):
        return {"error": f"Results file '{results_json_path}' not found."}

    with open(results_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine tasks list
    if "per_agent_details" in data and agent_name:
        if agent_name not in data["per_agent_details"]:
            available = list(data["per_agent_details"].keys())
            return {"error": f"Agent '{agent_name}' not found in results. Available: {available}"}
        tasks_data = data["per_agent_details"][agent_name]
    elif "tasks" in data:
        tasks_data = data["tasks"]
    elif "per_agent_details" in data:
        # Pick the first agent if not specified
        first_agent = list(data["per_agent_details"].keys())[0]
        tasks_data = data["per_agent_details"][first_agent]
        agent_name = first_agent
    else:
        tasks_data = []

    failed_tasks = [t for t in tasks_data if not t.get("success", False)]

    failure_by_type = Counter()
    failure_by_difficulty = Counter()
    failure_by_category = Counter()

    case_studies = []

    for t in failed_tasks:
        task_id = t.get("task_id")
        task_obj = get_task_by_id(task_id) if task_id else None

        ftype = t.get("failure_type", "unknown")
        failure_by_type[ftype] += 1

        diff = t.get("difficulty") or (task_obj.difficulty if task_obj else 0)
        failure_by_difficulty[f"L{diff}"] += 1

        cat = t.get("category") or (task_obj.category.value if task_obj else "unknown")
        failure_by_category[cat] += 1

        case_studies.append({
            "task_id": task_id,
            "category": cat,
            "difficulty": diff,
            "failure_type": ftype,
            "score": t.get("score"),
            "tool_calls": t.get("tool_calls", t.get("tool_call_count", 0)),
            "instruction": task_obj.instruction if task_obj else "",
            "notes": t.get("notes", ""),
            "failure_details": t.get("failure_details", {}),
        })

    analysis_report = {
        "agent": agent_name or data.get("agent_name", "unknown"),
        "total_tasks": len(tasks_data),
        "failed_count": len(failed_tasks),
        "failure_rate": round(len(failed_tasks) / len(tasks_data), 4) if tasks_data else 0.0,
        "by_type": dict(failure_by_type),
        "by_difficulty": dict(failure_by_difficulty),
        "by_category": dict(failure_by_category),
        "case_studies": case_studies,
    }

    # Determine file paths
    prefix = output_prefix or ("failure_analysis" + (f"_{agent_name}" if agent_name else ""))
    json_path = os.path.join(out_dir, f"{prefix}.json") if not output_prefix else (output_prefix + ".json" if not output_prefix.endswith(".json") else output_prefix)
    md_path = os.path.join(out_dir, f"{prefix}.md") if not output_prefix else (output_prefix + ".md" if not output_prefix.endswith(".md") else output_prefix)

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)

    # Save JSON report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_report, f, indent=2)

    # Save Markdown report
    agent_label = f" for Agent `{analysis_report['agent']}`" if agent_name else ""
    md_lines = [
        f"# Benchmark Failure Analysis & Trajectory Case Studies{agent_label}",
        "",
        f"**Total Tasks Evaluated:** {analysis_report['total_tasks']}  ",
        f"**Total Failures:** {analysis_report['failed_count']} ({analysis_report['failure_rate']*100:.1f}%)  ",
        "",
        "## Failure Breakdown by Taxonomy Type",
        "",
    ]

    if failure_by_type:
        for ftype, count in failure_by_type.items():
            md_lines.append(f"- **{ftype}**: {count} tasks")
    else:
        md_lines.append("No failures recorded! Perfect run.")

    md_lines.extend([
        "",
        "## Failure Breakdown by Difficulty",
        "",
    ])
    for diff, count in failure_by_difficulty.items():
        md_lines.append(f"- **{diff}**: {count} tasks")

    md_lines.extend([
        "",
        "## Trajectory Case Studies",
        "",
    ])

    for i, cs in enumerate(case_studies, 1):
        md_lines.extend([
            f"### Case Study {i}: {cs['task_id']} ({cs['failure_type']})",
            f"- **Category:** `{cs['category']}` | **Difficulty:** L{cs['difficulty']} | **Score:** {cs['score']} | **Tool Calls:** {cs['tool_calls']}",
            f"- **Instruction:** \"{cs['instruction']}\"",
            f"- **Root Cause Details:** {json.dumps(cs['failure_details'])}",
            f"- **Notes:** {cs['notes']}",
            "",
        ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Failure analysis complete! Saved to {json_path} and {md_path}")
    return analysis_report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze benchmark task failures.")
    parser.add_argument("--results", default="results/comparison/results.json", help="Path to results JSON file.")
    parser.add_argument("--agent", default=None, help="Agent name to filter by (for multi-agent comparison files).")
    parser.add_argument("--output", default=None, help="Output path prefix (without extension).")
    args = parser.parse_args()

    out_dir = os.path.dirname(args.output) if args.output else "results"
    output_prefix = os.path.basename(args.output) if args.output else None

    analyze_failures(
        results_json_path=args.results,
        out_dir=out_dir if out_dir else "results",
        agent_name=args.agent,
        output_prefix=args.output,
    )

