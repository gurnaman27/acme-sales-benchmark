"""Multi-Agent Comparison Experiment Runner.

Evaluates participant agents across all 30 benchmark tasks,
compiling performance metrics across families (Rule-based, Neural Net, LLM).
Generates comparison JSON, CSV, and formatted Markdown table with
per-difficulty and per-category breakdowns.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from acme.agents.baseline_simple import SimpleAgent
from acme.agents.mlp_agent import MLPAgent
from acme.agents.random_agent import RandomAgent
from acme.green_agent.runner import Runner
from acme.tasks.task_library import TASK_LIBRARY


def get_available_participant_agents() -> list[Any]:
    """Instantiate all available participant agents."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass

    agents = [
        SimpleAgent(),
        RandomAgent(seed=42),
        MLPAgent("standard"),
        MLPAgent("deep"),
        MLPAgent("residual"),
        MLPAgent("ensemble"),
    ]

    # Optionally add LLM agents if API keys are set

    try:
        from acme.agents.llm_agent import LLMAgent
        from acme.agents.providers.openai import OpenAIProvider
        if os.environ.get("OPENAI_API_KEY"):
            agents.append(LLMAgent(OpenAIProvider()))
    except Exception as e:
        print(f"Skipping OpenAIProvider: {e}")

    try:
        from acme.agents.llm_agent import LLMAgent
        from acme.agents.providers.deepseek import DeepSeekProvider
        if os.environ.get("DEEPSEEK_API_KEY"):
            agents.append(LLMAgent(DeepSeekProvider()))
    except Exception as e:
        print(f"Skipping DeepSeekProvider: {e}")

    return agents


def _classify_family(name: str) -> str:
    """Classify agent into architectural family."""
    if "simple" in name:
        return "Rule-Based"
    elif "random" in name:
        return "Random Baseline"
    elif "mlp" in name:
        return "Neural Net (MLP)"
    else:
        return "LLM"


def run_comparison_experiment(
    out_dir: str = "results/comparison",
    k: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full benchmark comparison across all participant agents.

    Returns:
        Structured dictionary containing summary and per-agent metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    agents = get_available_participant_agents()
    runner = Runner(seed=seed)

    summary_data = []
    per_agent_details = {}

    for agent in agents:
        print(f"Running benchmark for agent: {agent.name}...")
        res = runner.run_all(agent)

        family = _classify_family(agent.name)
        policy_violations = sum(t.policy_violations_introduced for t in res.results)

        # Per-difficulty breakdown
        by_diff = res.by_difficulty()
        diff_summary = {}
        for lvl, stats in sorted(by_diff.items()):
            diff_summary[f"L{lvl}"] = {
                "success_rate": round(stats["success_rate"], 4),
                "count": stats["count"],
                "successes": stats["successes"],
            }

        # Per-category breakdown
        by_cat = res.by_category()
        cat_summary = {}
        for cat, stats in by_cat.items():
            cat_summary[cat] = {
                "success_rate": round(stats["success_rate"], 4),
                "count": stats["count"],
                "successes": stats["successes"],
            }

        agent_entry = {
            "agent": agent.name,
            "family": family,
            "total_tasks": res.total_tasks,
            "success_count": sum(1 for t in res.results if t.success),
            "success_rate": round(res.success_rate, 4),
            "mean_score": round(res.mean_score, 4),
            "mean_tool_calls": round(res.mean_tool_calls, 2),
            "policy_violations": policy_violations,
            "by_difficulty": diff_summary,
            "by_category": cat_summary,
        }
        summary_data.append(agent_entry)

        # Per-task detail
        per_agent_details[agent.name] = [
            {
                "task_id": t.task_id,
                "success": t.success,
                "score": t.score,
                "tool_calls": t.tool_call_count,
                "failure_type": t.failure_type.value if t.failure_type else None,
            }
            for t in res.results
        ]

    # Save comparison JSON (full detail)
    full_output = {
        "benchmark_size": len(TASK_LIBRARY),
        "seed": seed,
        "agents_evaluated": len(summary_data),
        "summary": summary_data,
        "per_agent_details": per_agent_details,
    }
    json_path = os.path.join(out_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    # Save flat comparison CSV
    csv_path = os.path.join(out_dir, "results.csv")
    if summary_data:
        flat_keys = [
            "agent", "family", "total_tasks", "success_count",
            "success_rate", "mean_score", "mean_tool_calls", "policy_violations",
        ]
        # Add per-difficulty columns
        for lvl in ["L1", "L2", "L3", "L4"]:
            flat_keys.append(f"sr_{lvl}")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=flat_keys)
            writer.writeheader()
            for row in summary_data:
                flat_row = {k: row[k] for k in flat_keys[:8]}
                for lvl in ["L1", "L2", "L3", "L4"]:
                    flat_row[f"sr_{lvl}"] = row["by_difficulty"].get(lvl, {}).get("success_rate", 0.0)
                writer.writerow(flat_row)

    # Save Markdown Table
    md_path = os.path.join(out_dir, "comparison_table.md")
    md_lines = _generate_markdown_report(summary_data, seed)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"\nComparison complete! Outputs saved to {out_dir}")
    return full_output


def _generate_markdown_report(summary_data: list[dict], seed: int) -> list[str]:
    """Generate a rich Markdown comparison report."""
    lines = [
        "# Acme Sales Benchmark — Multi-Agent Comparison",
        "",
        f"**Total Benchmark Tasks:** {len(TASK_LIBRARY)}  ",
        f"**Random Seed:** {seed}  ",
        "",
        "## Overall Results",
        "",
        "| Agent | Family | Success Rate | Mean Score | Tool Calls | Policy Violations |",
        "|---|---|---|---|---|---|",
    ]

    for row in summary_data:
        lines.append(
            f"| `{row['agent']}` | {row['family']} "
            f"| {row['success_rate'] * 100:.1f}% "
            f"| {row['mean_score']:.3f} "
            f"| {row['mean_tool_calls']:.2f} "
            f"| {row['policy_violations']} |"
        )

    # Per-difficulty table
    lines.extend([
        "",
        "## Success Rate by Difficulty Level",
        "",
        "| Agent | L1 (Basic) | L2 (Multi-step) | L3 (Constraint) | L4 (Complex) |",
        "|---|---|---|---|---|",
    ])
    for row in summary_data:
        bd = row["by_difficulty"]
        lines.append(
            f"| `{row['agent']}` "
            f"| {bd.get('L1', {}).get('success_rate', 0) * 100:.0f}% "
            f"| {bd.get('L2', {}).get('success_rate', 0) * 100:.0f}% "
            f"| {bd.get('L3', {}).get('success_rate', 0) * 100:.0f}% "
            f"| {bd.get('L4', {}).get('success_rate', 0) * 100:.0f}% |"
        )

    # Per-category table
    lines.extend([
        "",
        "## Success Rate by Task Category",
        "",
    ])
    categories = set()
    for row in summary_data:
        categories.update(row["by_category"].keys())
    categories = sorted(categories)

    header = "| Agent | " + " | ".join(c.replace("_", " ").title() for c in categories) + " |"
    sep = "|---| " + " | ".join("---" for _ in categories) + " |"
    lines.append(header)
    lines.append(sep)
    for row in summary_data:
        cells = []
        for cat in categories:
            sr = row["by_category"].get(cat, {}).get("success_rate", 0)
            cells.append(f"{sr * 100:.0f}%")
        lines.append(f"| `{row['agent']}` | " + " | ".join(cells) + " |")

    return lines


if __name__ == "__main__":
    run_comparison_experiment()
