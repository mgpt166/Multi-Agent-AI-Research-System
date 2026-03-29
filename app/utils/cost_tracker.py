"""
app/utils/cost_tracker.py
=========================
Per-request cost tracking for all LLM calls in the research pipeline.

CostTracker is created once per research job and accumulates token usage
across all agents. It provides budget enforcement (hard stop) and a warning
threshold that signals the pipeline to reduce further spending.

Pricing is looked up dynamically per model name so mixed-provider pipelines
(e.g. Groq for research + Anthropic for the eval judge) are costed accurately.
The fallback price comes from TOKEN_PRICE_INPUT / TOKEN_PRICE_OUTPUT in config.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

_logger = logging.getLogger(__name__)

# ── Per-model pricing table (USD per million tokens) ──────────────────────────
# Add new models here as needed. Prefix-matched so "llama-3.1-8b-instant" hits
# the "llama-3.1-8b" entry. Last entry is the catch-all fallback.
_MODEL_PRICING: list[tuple[str, float, float]] = [
    # (model_prefix,                   input $/1M,  output $/1M)
    ("llama-3.1-8b",                   0.05,        0.08),
    ("llama-3.3-70b",                  0.59,        0.79),
    ("llama-3.1-70b",                  0.59,        0.79),
    ("mixtral-8x7b",                   0.27,        0.27),
    ("claude-haiku-4-5",               1.00,        5.00),
    ("claude-haiku",                   1.00,        5.00),
    ("claude-sonnet-4-6",              3.00,       15.00),
    ("claude-sonnet",                  3.00,       15.00),
    ("claude-opus-4-6",                5.00,       25.00),
    ("claude-opus",                    5.00,       25.00),
    ("gpt-4o-mini",                    0.15,        0.60),
    ("gpt-4o",                         2.50,       10.00),
    ("gpt-3.5",                        0.50,        1.50),
]


def _fallback_prices() -> tuple[float, float]:
    from app.config import TOKEN_PRICE_INPUT, TOKEN_PRICE_OUTPUT
    return TOKEN_PRICE_INPUT, TOKEN_PRICE_OUTPUT


def get_model_pricing(model: str) -> tuple[float, float]:
    """
    Return (input_price_per_million, output_price_per_million) for a model.

    Matches by prefix (case-insensitive). Falls back to TOKEN_PRICE_INPUT /
    TOKEN_PRICE_OUTPUT from config if no match is found.

    Args:
        model: Model name string (e.g. "llama-3.3-70b-versatile").

    Returns:
        Tuple of (input $/1M tokens, output $/1M tokens).
    """
    model_lower = model.strip().lower()
    for prefix, inp, out in _MODEL_PRICING:
        if model_lower.startswith(prefix.lower()):
            return inp, out
    # Fallback to config values
    from app.config import TOKEN_PRICE_INPUT, TOKEN_PRICE_OUTPUT
    _logger.debug("No pricing entry for model '%s' — using config defaults", model)
    return TOKEN_PRICE_INPUT, TOKEN_PRICE_OUTPUT


@dataclass
class AgentUsage:
    """Token usage record for a single agent call."""
    agent_name: str
    model: str
    input_tokens: int
    output_tokens: int
    cost: float


class CostTracker:
    """
    Tracks cumulative token usage and cost for a single research job.

    Each add_usage() call specifies the model used so pricing is looked up
    dynamically — accurate even when multiple models are used in one job.

    Args:
        max_budget:        Hard ceiling in USD — pipeline stops if exceeded.
        warning_threshold: USD amount at which a warning is logged.
    """

    def __init__(self, max_budget: float, warning_threshold: float) -> None:
        self._max_budget = max_budget
        self._warning_threshold = warning_threshold
        self._usages: list[AgentUsage] = []
        self._warning_triggered = False
        self._budget_exceeded = False

    def add_usage(
        self,
        input_tokens: int,
        output_tokens: int,
        agent_name: str,
        model: str = "",
    ) -> None:
        """
        Record token usage from one LLM API call.

        Args:
            input_tokens:  Prompt tokens consumed.
            output_tokens: Completion tokens generated.
            agent_name:    Identifier for the calling agent (e.g. "lead_researcher_plan").
            model:         Model name used for this call — drives dynamic pricing lookup.
        """
        inp_price, out_price = get_model_pricing(model) if model else _fallback_prices()
        cost = (input_tokens * inp_price / 1_000_000) + (output_tokens * out_price / 1_000_000)
        self._usages.append(
            AgentUsage(
                agent_name=agent_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost,
            )
        )
        if self.is_warning_threshold() and not self._warning_triggered:
            _logger.warning(
                "Cost warning threshold reached after '%s': total=%.4f threshold=%.4f",
                agent_name, self.total_cost, self._warning_threshold,
            )
            self._warning_triggered = True
        if self.is_budget_exceeded() and not self._budget_exceeded:
            _logger.error(
                "Budget exceeded after '%s' call: total=%.4f max=%.4f",
                agent_name, self.total_cost, self._max_budget,
            )
            self._budget_exceeded = True

    @property
    def total_cost(self) -> float:
        """Current cumulative cost in USD."""
        return sum(u.cost for u in self._usages)

    @property
    def total_input_tokens(self) -> int:
        """Total input tokens used across all calls."""
        return sum(u.input_tokens for u in self._usages)

    @property
    def total_output_tokens(self) -> int:
        """Total output tokens generated across all calls."""
        return sum(u.output_tokens for u in self._usages)

    @property
    def remaining_budget(self) -> float:
        """How much USD budget remains before the hard ceiling."""
        return max(0.0, self._max_budget - self.total_cost)

    def is_budget_exceeded(self) -> bool:
        """Return True if the hard cost ceiling has been reached."""
        return self.total_cost >= self._max_budget

    def is_warning_threshold(self) -> bool:
        """Return True if the warning threshold has been reached."""
        return self.total_cost >= self._warning_threshold

    def get_summary(self) -> dict:
        """
        Return full cost breakdown suitable for API responses and job_store.

        Returns:
            dict with total cost, per-agent breakdown, budget status.
        """
        breakdown: dict[str, dict] = {}
        for u in self._usages:
            if u.agent_name in breakdown:
                entry = breakdown[u.agent_name]
                entry["input_tokens"] += u.input_tokens
                entry["output_tokens"] += u.output_tokens
                entry["cost"] += u.cost
            else:
                breakdown[u.agent_name] = {
                    "model": u.model,
                    "input_tokens": u.input_tokens,
                    "output_tokens": u.output_tokens,
                    "cost": u.cost,
                }
        # Round cost values for clean JSON output
        for entry in breakdown.values():
            entry["cost"] = round(entry["cost"], 6)

        return {
            "total": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "breakdown": breakdown,
            "budget_remaining": round(self.remaining_budget, 6),
            "budget_exceeded": self.is_budget_exceeded(),
            "warning_triggered": self._warning_triggered,
        }
