# Acme Sales Benchmark — Comprehensive Research Report

**Authors:** Acme AI Evaluation Team  
**Date:** September 2026  
**Repository:** `acme-sales-benchmark`  

---

## 1. Abstract

Evaluating autonomous AI agents in enterprise domain environments requires multi-faceted benchmarks that go beyond static single-turn QA to assess complex multi-step tool call planning, policy constraint compliance, and state mutation correctness. In this work, we present the **Acme Sales Benchmark**, a standardized simulation framework featuring **30 tasks across 4 difficulty levels** and **5 CRM functional categories** (Opportunity Management, Follow-up Tasks, Product Recommendation, Policy Conflict Resolution, and Scheduling). We evaluate 7 participant agents across 4 architectural families: hardcoded rule-based baselines, random baselines, learned multi-layer perceptron (MLP) neural networks (standard, deep, residual, and ensemble variants), and frontier large language model (LLM) agents (OpenAI GPT-4o-mini and DeepSeek Chat). Our empirical findings demonstrate that while foundation LLMs outperform neural network models on multi-step reasoning, architectural depth and residual skip connections enable specialized MLPs to match or exceed baseline LLMs on specific constraint validation tasks.

---

## 2. Problem

Enterprise Customer Relationship Management (CRM) automation demands high reliability, policy compliance, and strict adherence to organizational rules (such as max allowable discount thresholds, rep assignment boundaries, and working hour schedules). Existing benchmark suites often evaluate code execution or general web browsing in unconstrained environments, failing to capture the structured business policy constraints and relational state consistency inherent in enterprise CRM platforms.

---

## 3. Motivation

Building autonomous agents for enterprise sales workflows requires evaluating:
1. **Tool Use & Parameter Accuracy:** Does the agent select valid tools and pass correctly structured parameters?
2. **Multi-Step Task Planning:** Can the agent sequence dependent operations (e.g., querying rep availability before scheduling a meeting)?
3. **Policy & Constraint Compliance:** Does the agent honor organizational rules (e.g., declining policy-violating discount requests)?
4. **Architectural Scaling:** How do lightweight neural network policies compare against large foundation language models in domain-specific tasks?

---

## 4. Related Work

- **ToolBench & WebArena:** Benchmark suites assessing LLM tool-use in web/API environments.
- **TAU-Bench:** Recent benchmarks measuring domain-specific multi-turn user interaction and system tool use.
- **Imitation Learning for Tool Use:** Training compact neural net policy models on expert trajectory datasets to perform lightweight local tool calling without full LLM inference overhead.

---

## 5. Benchmark Design

The Acme Sales Benchmark consists of **30 curated tasks** structured into 4 difficulty tiers:
- **Level 1 (Basic, 8 tasks):** Single-step tool calls (e.g., look up customer record, fetch policy text).
- **Level 2 (Multi-step, 7 tasks):** 2–3 dependent tool calls (e.g., retrieve customer opportunity and update stage).
- **Level 3 (Constraint & Policy, 8 tasks):** Evaluating actions against business policies (e.g., rejecting an out-of-bounds discount).
- **Level 4 (Complex End-to-End, 7 tasks):** Multi-entity workflows involving customers, opportunities, reps, and calendar schedules.

---

## 6. Environment & Architecture

The simulated environment models an enterprise CRM system with 5 core entity models:
- **Customers:** Account tier, status, notes, history.
- **Products:** Pricing, eligibility rules, categories.
- **Opportunities:** Value, stage, assigned rep, close date.
- **Meetings & Schedules:** Rep calendar availability, slots, working hours.
- **Policies:** Business rules for discounts, eligibility, and scheduling.

The environment exposes **20 deterministic tools** via a unified `ToolRegistry`.

---

## 7. Task Design & Schema

Every task follows a strict specification (`BenchmarkTask`):
- `task_id`: Unique identifier (`T001` to `T030`).
- `instruction`: Natural language instruction.
- `category`: Functional category (`opportunity_mgmt`, `follow_up`, `product_recommendation`, `policy_conflict`, `scheduling`).
- `difficulty`: Integer level (1 to 4).
- `required_tools`: Expected tool set.
- `forbidden_tools`: Disallowed tool set.
- `evaluator_spec`: Multi-component evaluation criteria.

---

## 8. Evaluation Methodology

The evaluator uses a multi-component scoring function:

$$\text{Score} = w_{\text{state}} S_{\text{state}} + w_{\text{tools}} S_{\text{tools}} + w_{\text{policy}} S_{\text{policy}} + w_{\text{eff}} S_{\text{eff}} + w_{\text{forb}} S_{\text{forb}}$$

With redistributed weights for strict floor calibration:
- $w_{\text{state}} = 0.50$ (State Correctness)
- $w_{\text{tools}} = 0.25$ (Required Tools)
- $w_{\text{policy}} = 0.15$ (Policy Compliance)
- $w_{\text{eff}} = 0.05$ (Efficiency Score)
- $w_{\text{forb}} = 0.05$ (No Forbidden Tools)

A task succeeds ($\text{Success} = \text{True}$) if and only if $\text{Score} \ge 0.75$ and no policy violations were introduced.

---

## 9. Experimental Setup

We evaluate **7 participant agents across 4 architectural families**:
1. **Rule-Based:** `simple_rule_based` (Hardcoded expert heuristic per task)
2. **Random Baseline:** `random_baseline` (Uniform random tool selection)
3. **Neural Net (MLP):**
   - `mlp_standard`: 2-layer MLP (128-64 units)
   - `mlp_deep`: 4-layer MLP (256-128-64-32 units)
   - `mlp_residual`: 4-layer MLP with residual skip connections
   - `mlp_ensemble`: Majority-vote ensemble of 3 MLPs
4. **Large Language Models:**
   - `llm_openai_gpt-4o-mini`: Closed-source frontier model
   - `llm_deepseek_deepseek-chat`: Open-weight architecture model

---

## 10. Results & Multi-Agent Comparison

*(Updated automatically upon benchmark execution completion)*

### Overall Results Table

| Agent | Family | Success Rate | Mean Score | Tool Calls | Policy Violations |
|---|---|:---:|:---:|:---:|:---:|
| `simple_rule_based` | Rule-Based | **100.0%** | **1.000** | 1.90 | 0 |
| `llm_deepseek_deepseek-chat` | LLM | **96.7%** | **0.990** | 4.30 | 0 |
| `mlp_deep` | Neural Net (MLP) | **86.7%** | **0.947** | 1.80 | 0 |
| `mlp_residual` | Neural Net (MLP) | **83.3%** | **0.947** | 1.77 | 0 |
| `llm_openai_gpt-4o-mini` | LLM | **80.0%** | **0.952** | 2.33 | 0 |
| `mlp_ensemble` | Neural Net (MLP) | **60.0%** | 0.834 | 1.37 | 2 |
| `mlp_standard` | Neural Net (MLP) | **56.7%** | 0.860 | 1.37 | 2 |
| `random_baseline` | Random Baseline | **3.3%** | 0.523 | 2.83 | 1 |

---

## 11. Failure Taxonomy Analysis

Failures across non-rule-based agents fall into three primary taxonomy categories:
1. **Planning Failures:** Invoking downstream tools before querying prerequisite state.
2. **Execution Failures:** Invalid argument payload formatting during CRM updates.
3. **Tool Use / Policy Compliance Failures:** Parameter selections exceeding policy thresholds.

---

## 12. Limitations

- Evaluation is performed in a simulated environment with synthetic customer data.
- Rate limits and transient API latency affect live LLM evaluation speed.

---

## 13. Future Work

- Extend state evaluation to real-world multi-tenant CRM database sandboxes.
- Benchmark fine-tuned open-source local LLMs (Llama 3 / Mistral) against enterprise tasks.

---

## 14. Conclusion

The Acme Sales Benchmark provides a rigorous framework for assessing agent tool calling, planning, and policy compliance. Deep learning policy networks with residual skip connections present a viable, low-latency alternative to full LLM reasoning for domain-constrained workflows.
