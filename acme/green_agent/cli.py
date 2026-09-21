"""Command-line entry point: `run-benchmark`.

Usage:
    run-benchmark --agent simple --tasks all
    run-benchmark --agent random --tasks all
    run-benchmark --agent mlp_standard --tasks all
    run-benchmark --agent llm_openai --tasks T001,T002
    run-benchmark --compare-all
    run-benchmark --validate
    run-benchmark --train-mlp
    run-benchmark --failure-analysis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from acme.agents.agent_interface import ParticipantAgent
from acme.agents.baseline_simple import SimpleAgent
from acme.agents.mlp_agent import MLPAgent
from acme.agents.random_agent import RandomAgent
from acme.green_agent.export import (
    export_reliability_json,
    export_results_csv,
    export_results_json,
)
from acme.green_agent.runner import Runner
from acme.tasks.task_library import TASK_LIBRARY


_AVAILABLE_AGENTS = [
    "simple", "random",
    "mlp_standard", "mlp_deep", "mlp_residual", "mlp_ensemble",
    "llm_openai", "llm_deepseek", "llm_compatible",
]


def _build_agent(name: str) -> ParticipantAgent:
    """Instantiate a participant agent by name."""
    if name == "simple":
        return SimpleAgent()
    elif name == "random":
        return RandomAgent(seed=42)
    elif name.startswith("mlp_"):
        variant = name.split("_", 1)[1]
        if variant in ("standard", "deep", "residual", "ensemble"):
            return MLPAgent(variant=variant)
    elif name == "llm_openai":
        from acme.agents.llm_agent import LLMAgent
        from acme.agents.providers.openai import OpenAIProvider
        return LLMAgent(OpenAIProvider())
    elif name == "llm_deepseek":
        from acme.agents.llm_agent import LLMAgent
        from acme.agents.providers.deepseek import DeepSeekProvider
        return LLMAgent(DeepSeekProvider())
    elif name == "llm_compatible":
        from acme.agents.llm_agent import LLMAgent
        from acme.agents.providers.openai_compatible import OpenAICompatibleProvider
        return LLMAgent(OpenAICompatibleProvider())

    raise ValueError(
        f"Unknown agent '{name}'. Available: {', '.join(_AVAILABLE_AGENTS)}"
    )


def _select_tasks(spec: str):
    if spec == "all":
        return TASK_LIBRARY
    ids = {x.strip() for x in spec.split(",") if x.strip()}
    selected = [t for t in TASK_LIBRARY if t.task_id in ids]
    missing = ids - {t.task_id for t in selected}
    if missing:
        raise ValueError(f"Unknown task IDs: {sorted(missing)}")
    return selected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run-benchmark",
        description="Run the Acme Sales benchmark against a participant agent.",
    )
    parser.add_argument(
        "--agent", default="simple",
        help=f"Agent name. Available: {', '.join(_AVAILABLE_AGENTS)}. Default: simple",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for state generation. Default: 42",
    )
    parser.add_argument(
        "--tasks", default="all",
        help="'all' or comma-separated task IDs. Default: all",
    )
    parser.add_argument(
        "--k", type=int, default=1,
        help="Repetitions per task for reliability (pass@k). Default: 1",
    )
    parser.add_argument(
        "--output", default=None,
        help="Directory to write results.json and results.csv.",
    )
    parser.add_argument(
        "--compare-all", action="store_true",
        help="Run multi-agent comparison experiment across all available agents.",
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="Run evaluator validation suite against 6 synthetic agents.",
    )
    parser.add_argument(
        "--train-mlp", action="store_true",
        help="Train all 4 MLP variants (standard, deep, residual, ensemble) via imitation learning.",
    )
    parser.add_argument(
        "--failure-analysis", action="store_true",
        help="Run failure analysis on saved results.json file.",
    )
    args = parser.parse_args(argv)

    if args.compare_all:
        from acme.experiments.run_comparison import run_comparison_experiment
        out_dir = args.output or "results/comparison"
        run_comparison_experiment(out_dir=out_dir, k=args.k, seed=args.seed)
        return 0

    if args.validate:
        from acme.validation.validation_suite import run_evaluator_validation_suite
        rep = run_evaluator_validation_suite(seed=args.seed)
        print("\nEvaluator Validation Suite Results:")
        all_calibrated = True
        for k, v in rep.items():
            status = "✓" if v['calibrated'] else "✗"
            print(f"  {status} {k:28s}: Score={v['mean_score']:.4f} | Band={v['expected_band']} | Calibrated={v['calibrated']}")
            if not v['calibrated']:
                all_calibrated = False
        print(f"\n{'All agents calibrated!' if all_calibrated else 'CALIBRATION FAILURE detected.'}")
        return 0 if all_calibrated else 1

    if args.train_mlp:
        from acme.agents.imitation_trainer import train_mlp_agent
        print("Training MLP neural network variants via imitation learning...")
        for v in ["standard", "deep", "residual", "ensemble"]:
            p = train_mlp_agent(v, epochs=30)
            print(f"  ✓ Trained {v:10s} → {p}")
        print("\nAll MLP variants trained successfully.")
        return 0

    if args.failure_analysis:
        from acme.experiments.failure_analysis import analyze_failures
        input_file = args.output or "results/results.json"
        analyze_failures(results_json_path=input_file)
        return 0

    try:
        agent = _build_agent(args.agent)
        tasks = _select_tasks(args.tasks)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    runner = Runner(seed=args.seed)

    print("=" * 60)
    print(f"ACME SALES BENCHMARK — Agent: {agent.name}")
    print("=" * 60)
    print(f"Tasks: {len(tasks)}  |  Seed: {args.seed}  |  k={args.k}")
    print()

    if args.k > 1:
        # Reliability mode
        reliability = runner.run_all_with_reliability(agent, k=args.k, tasks=tasks)
        pass_at_1 = sum(1 for r in reliability if r.pass_at_1) / len(reliability)
        pass_at_k = sum(1 for r in reliability if r.pass_at_k) / len(reliability)
        consistency = sum(r.consistency for r in reliability) / len(reliability)

        print(f"pass@1       : {pass_at_1:.1%}")
        print(f"pass@{args.k}       : {pass_at_k:.1%}")
        print(f"consistency  : {consistency:.1%}")

        if args.output:
            out = Path(args.output)
            out.mkdir(parents=True, exist_ok=True)
            export_reliability_json(reliability, out / "reliability.json")
            print(f"\nWrote {out / 'reliability.json'}")

        # Also run one deterministic pass for the full report
        result = runner.run_all(agent, tasks=tasks)
    else:
        result = runner.run_all(agent, tasks=tasks)

    print()
    print(f"Success      : {result.successful_tasks}/{result.total_tasks} "
          f"({result.success_rate:.1%})")
    print(f"Mean score   : {result.mean_score:.3f}")
    print(f"Mean toolcalls: {result.mean_tool_calls:.2f}")

    print("\nBy category:")
    for cat, stats in result.by_category().items():
        print(f"  {cat:24s} {stats['successes']}/{stats['count']} "
              f"({stats['success_rate']:.1%})")

    print("\nBy difficulty:")
    for lvl, stats in sorted(result.by_difficulty().items()):
        print(f"  L{lvl}  {stats['successes']}/{stats['count']} "
              f"({stats['success_rate']:.1%})")

    if result.failure_breakdown():
        print("\nFailures:")
        for ft, count in sorted(result.failure_breakdown().items()):
            print(f"  {ft:20s} {count}")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        export_results_json(result, out / "results.json")
        export_results_csv(result, out / "results.csv")
        print(f"\nWrote {out / 'results.json'}")
        print(f"Wrote {out / 'results.csv'}")

    return 0 if result.success_rate == 1.0 else 1


if __name__ == "__main__":
    raise SystemExit(main())