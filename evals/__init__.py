"""
evals/
======
Evaluation framework for the Multi-Agent AI Research System.

Measures research quality using an LLM-as-judge approach:
    - A pool of test cases across 4 complexity tiers
    - A weighted rubric covering accuracy, citations, completeness, etc.
    - Claude as judge — scores each research output against the rubric
    - A CLI runner that executes test cases and saves results to JSON

Entry point:
    python scripts/run_eval.py

See evals/runner.py for the orchestration logic.
"""
