"""Post-Training Module for Benchmark Agents.

Provides:
  - Supervised Fine-Tuning (SFT) on multi-source success trajectories
  - OpenAI fine-tuning JSONL exporter
  - Iterative Rejection Sampling / Self-Taught Reasoner (STaR) loop
  - Preference-based DPO optimization
"""
