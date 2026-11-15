"""Token usage and cost, from the numbers the API returns with every response.

K2 [docs]: cost comes from the API's own usage fields, cache reads and writes included, never
  from an estimate. Check these prices against the pricing page before quoting any cost.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# USD per million tokens: (input, output, cache read). A 5-minute cache write costs 1.25x input.
PRICES = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.10),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),  # prompts up to 100K tokens; more is billed higher
}
CACHE_WRITE_MULTIPLIER = 1.25


def check_priced(model: str) -> None:
    """Fail before the first call, not after it: an unpriced model would be billed untracked."""
    if model not in PRICES:
        raise ValueError(f"No price for {model!r}. Add it to PRICES in agentic_evals/usage.py.")


@dataclass
class Usage:
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    calls: int = 0

    def add(self, usage: Any) -> None:
        # One known gap: when a refusal falls back to another model, the top-level usage covers
        #   only the attempt that answered, and it's priced here at this Usage's model. The
        #   response's usage.iterations has the exact per-attempt figures.
        self.calls += 1
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.cache_read_tokens += usage.cache_read_input_tokens or 0
        self.cache_write_tokens += usage.cache_creation_input_tokens or 0

    @property
    def cost_usd(self) -> float:
        p_in, p_out, p_read = PRICES[self.model]
        return (self.input_tokens * p_in + self.output_tokens * p_out
                + self.cache_read_tokens * p_read
                + self.cache_write_tokens * p_in * CACHE_WRITE_MULTIPLIER) / 1e6

    @property
    def cache_hit_rate(self) -> float:
        """K3 [docs]: share of input read from cache. Near zero across calls means the prefix
        is changing between calls; see C3 in agent.py."""
        total = self.input_tokens + self.cache_read_tokens + self.cache_write_tokens
        return self.cache_read_tokens / total if total else 0.0
