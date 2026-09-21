# Acme Sales Benchmark: Evaluating AI Agents on Multi-Step Enterprise Sales Workflows

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-229%20passed-green.svg)](tests/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

The **Acme Sales Benchmark** is a benchmark for evaluating autonomous AI agents on multi-step enterprise sales workflows. The benchmark features a synthetic, stateful CRM environment, 21 granular CRM tools, 7 enterprise business safety policies, an anti-gaming composite evaluator with failure taxonomy, an 8-agent architectural roster (Rule-Based, Random, 4 Neural Net MLPs, 2 Foundation LLMs), a post-training self-improvement pipeline (SFT, STaR, DPO), and a standard Agent-to-Agent (A2A) protocol server.

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and set up a Python virtual environment:

```bash
git clone https://github.com/acme/acme-sales-benchmark.git
cd acme-sales-benchmark

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Run the Unit Test Suite

```bash
pytest tests/ -v
```

### 3. Run the Multi-Agent Benchmark Comparison

```bash
python -m acme.experiments.run_comparison
```

### 4. Run Reliability & pass@k Experiments

```bash
python -m acme.experiments.run_reliability
```

### 5. Run Post-Training & Self-Improvement Experiments

```bash
python -m acme.experiments.run_post_training_comparison
```

### 6. Launch the A2A Server

```bash
uvicorn acme.green_agent.a2a_server:app --host 0.0.0.0 --port 9009
```

---

## 📂 Repository Structure

```
acme-sales-benchmark/
├── acme/                           # Core Benchmark Package
│   ├── agents/                     # Agent Roster & Models
│   │   ├── baseline_simple.py      # Rule-Based Baseline (SimpleAgent)
│   │   ├── random_agent.py         # Lower-bound Random Baseline
│   │   ├── mlp_agent.py            # Neural Net Agent (4 MLP variants)
│   │   ├── mlp_models.py           # Pure NumPy MLPs (Standard, Deep, Residual, Ensemble)
│   │   ├── feature_encoder.py      # 64-dim State & Action Featurizer
│   │   ├── imitation_trainer.py    # Imitation Learning Trainer
│   │   ├── llm_agent.py            # ReAct / Tool-calling LLM Agent
│   │   ├── providers/              # LLM Provider Backends (OpenAI, DeepSeek)
│   │   └── post_training/          # Post-Training & Self-Improvement
│   │       ├── collect_successes.py # Multi-source Dataset Collector & JSONL Exporter
│   │       ├── sft_trainer.py      # Supervised Fine-Tuning (SFT)
│   │       ├── iterative_sft.py    # Self-Taught Reasoner (STaR) Loop
│   │       └── dpo_trainer.py      # Direct Preference Optimization (DPO & Warm-Start)
│   ├── environment/                # Synthetic Data & State Management
│   │   ├── data_generator.py       # Seeded CRM Data Generator
│   │   ├── state.py                # Pydantic State & Collection Models
│   │   └── policies.py             # 7 Enterprise Policy Audit Rules
│   ├── evaluation/                 # Evaluator & Failure Taxonomy
│   │   ├── evaluator.py            # Anti-gaming Composite Evaluator
│   │   ├── metrics.py              # TaskResult & BenchmarkResult Models
│   │   └── failure_taxonomy.py     # 10-class Failure Classifier
│   ├── green_agent/                # Benchmark Runner & A2A Server
│   │   ├── runner.py               # Green Agent Benchmark Execution Engine
│   │   └── a2a_server.py           # FastAPI Agent-to-Agent Protocol Server
│   ├── tasks/                      # 30 Benchmark Tasks (L1-L4)
│   │   ├── task_schema.py          # BenchmarkTask & ExpectedOutcome Schemas
│   │   └── task_library.py         # 30 Calibrated Tasks & Patches
│   ├── tools/                      # 21 Stateful CRM Tools
│   │   ├── base_tool.py            # BaseTool & ToolCallResult Schemas
│   │   └── tool_registry.py        # CRM Tools & Execution Engine
│   └── cli.py                      # Command Line Interface
├── data/                           # Exported Datasets (OpenAI JSONL)
├── models/                         # Saved Neural Policy Weights (.npz)
├── results/                        # Evaluation Reports & Comparison Data
│   ├── comparison/                 # 8-Agent Benchmark Comparison Reports
│   ├── reliability/                # Reliability & pass@k Reports
│   └── post_training/              # Post-Training Empirical Reports
├── tests/                          # 229 Unit Tests
├── pyproject.toml                  # Project Metadata & Dependencies
└── README.md                       # Project Documentation
```

---

## 🛠️ 21 Stateful CRM Tools Specification

The environment exposes 21 granular, stateful API tools across 6 categories:

| Category | Tool | Description |
|---|---|---|
| **Customer (4)** | `search_customers` | Query customers by industry, tier, or territory |
| | `get_customer` | Fetch customer profile, tier, and assigned sales rep |
| | `get_customer_history` | Retrieve complete activity interaction history |
| | `update_customer` | Modify customer notes, tier, or assigned sales rep |
| **Product (3)** | `search_products` | Search product catalog by min tier or price |
| | `get_product` | Fetch product specifications, base price, and min tier |
| | `check_product_eligibility` | Verify customer eligibility for product purchase |
| **Opportunity (4)** | `get_opportunity` | Fetch opportunity details by ID |
| | `search_opportunities` | Search opportunities by stage, customer ID, or value |
| | `update_opportunity` | Update opportunity stage, value, or next action |
| | `create_opportunity` | Open a new sales deal opportunity |
| **Calendar (5)** | `get_current_time` | Fetch current simulation timestamp and day of week |
| | `get_available_slots` | Query rep calendar availability within time window |
| | `schedule_meeting` | Book a customer calendar meeting |
| | `cancel_meeting` | Cancel an existing meeting with reason |
| | `get_meetings` | Retrieve scheduled calendar meetings |
| **Followup (3)** | `create_followup` | Schedule a follow-up task with due date |
| | `get_followups` | Retrieve pending follow-up task items |
| | `complete_followup` | Mark follow-up task as completed |
| **Policy (2)** | `check_policy` | Audit specific rule compliance |
| | `get_policies` | Retrieve enterprise business policy definitions |

---

## 🎯 Benchmark Tasks (30 Tasks Across L1–L4)

The benchmark evaluates agents across 30 tasks spanning 4 difficulty levels:

- **L1 Basic (8 Tasks)**: Single-tool read/query operations (e.g. `T001`–`T008`).
- **L2 Multi-step (7 Tasks)**: Multi-tool read-modify-update workflows with state dependencies (e.g. `T009`–`T015`).
- **L3 Constraint/Refusal (8 Tasks)**: Tasks testing policy validation, conditional refusals, and threshold checks (e.g. `T016`–`T023`).
- **L4 Complex Multi-System (7 Tasks)**: 4+ step cross-collection orchestration tasks (e.g. `T024`–`T030`).

---

## 📊 Composite Evaluator & Anti-Gaming Design

The evaluator scores task runs **strictly from ground-truth state changes and tool call logs**, ignoring agent natural language text to prevent prompt injection or gaming.

$$\text{Score} = 0.50 \cdot S_{\text{state}} + 0.25 \cdot S_{\text{tools}} + 0.15 \cdot S_{\text{policy}} + 0.05 \cdot S_{\text{efficiency}} + 0.05 \cdot S_{\text{forbidden}}$$

---

## 📈 Multi-Agent Benchmark Results (8 Agents)

Evaluated across 30 tasks with seed 42:

| Rank | Agent Name | Family | Success Rate | Mean Score | Mean Tool Calls | Policy Violations |
|:---:|---|---|:---:|:---:|:---:|:---:|
| 1 | `simple_rule_based` | Rule-Based | **100.0%** | **1.000** | 1.90 | 0 |
| 2 | `llm_deepseek_deepseek-chat` | LLM | **96.7%** | **0.990** | 4.30 | 0 |
| 3 | `mlp_deep` | Neural Net (MLP) | **86.7%** | **0.947** | 1.80 | 0 |
| 4 | `mlp_residual` | Neural Net (MLP) | **83.3%** | **0.947** | 1.77 | 0 |
| 5 | `llm_openai_gpt-4o-mini` | LLM | **80.0%** | **0.952** | 2.33 | 0 |
| 6 | `mlp_ensemble` | Neural Net (MLP) | **60.0%** | 0.834 | 1.37 | 2 |
| 7 | `mlp_standard` | Neural Net (MLP) | **56.7%** | 0.860 | 1.37 | 2 |
| 8 | `random_baseline` | Random Baseline | **3.3%** | 0.523 | 2.83 | 1 |

---

## 🔁 Reliability & pass@k Performance ($k=3$)

| Agent | pass@1 Rate | pass@k Rate | Consistency | Mean Score | Score StdDev |
|---|:---:|:---:|:---:|:---:|:---:|
| `simple_rule_based` | **100.0%** | **100.0%** | 100.0% | 1.000 | 0.0000 |
| `mlp_deep` | **86.7%** | **86.7%** | 86.7% | 0.947 | 0.0000 |
| `mlp_deep_sft` | **90.0%** | **90.0%** | 90.0% | 0.957 | 0.0000 |

Deterministic policies (`simple_rule_based`, `mlp_deep`, `mlp_deep_sft`) exhibit **100% consistency (0.0000 StdDev)** across repeated runs.

---

## 🔬 Post-Training & Self-Improvement Empirical Results

We evaluated 4 post-training paradigms:

| Post-Training Paradigm | Agent Model | Success Rate | Mean Score | Mean Tool Calls | L1 | L2 | L3 | L4 |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `mlp_deep (Imitation Baseline)` | `mlp_deep` | **86.7%** | 0.947 | 1.80 | 100% | 57% | 100% | 86% |
| **`mlp_deep_sft (SFT Augmented)`** | `mlp_deep` | **90.0%** | **0.957** | 1.80 | 100% | 57% | 100% | **100%** |
| **`mlp_residual_sft (SFT Residual)`** | `mlp_residual` | **90.0%** | **0.957** | 1.80 | 100% | 57% | 100% | **100%** |
| `mlp_deep_star (STaR Iterative)` | `mlp_deep` | **83.3%** | 0.932 | 1.70 | 100% | 57% | 100% | 71% |
| `mlp_deep_dpo (Baseline DPO)` | `mlp_deep` | **83.3%** | 0.947 | 2.43 | 100% | 57% | 88% | 86% |
| **`mlp_deep_dpo_warm (Warm-Start DPO)`** | `mlp_deep` | **90.0%** | **0.957** | 1.80 | 100% | 57% | 100% | **100%** |

---

## 🌐 Agent-to-Agent (A2A) Protocol Server

The benchmark provides a production-grade FastAPI server adhering to the Agent-to-Agent (A2A) protocol specification:

- `GET /.well-known/agent-card.json` — Returns agent capability metadata.
- `GET /health` — Server health check endpoint.
- `POST /run` — Benchmark execution engine endpoint.

### Terminal Demonstration:

```bash
# Start server in background
uvicorn acme.green_agent.a2a_server:app --port 9009 &

# Query Agent Card Metadata
curl -s http://localhost:9009/.well-known/agent-card.json | jq .

# Execute Task Run via A2A Protocol
curl -s -X POST http://localhost:9009/run \
  -H "Content-Type: application/json" \
  -d '{"agent_name": "simple", "task_ids": ["T001", "T002"]}' | jq .
```

---

## 🧪 Test Suite & Verification

The codebase includes **229 unit tests** with 100% pass rate:

```bash
python -m pytest tests/ -q
```

---

## 📖 Citation & License

```bibtex
@article{acme_sales_benchmark_2026,
  title={Acme Sales Benchmark: Evaluating AI Agents on Multi-Step Enterprise Sales Workflows},
  author={Acme AI Research Team},
  year={2026},
  journal={Internal Agent Benchmark Evaluation}
}
```

This project is licensed under the [MIT License](LICENSE).
