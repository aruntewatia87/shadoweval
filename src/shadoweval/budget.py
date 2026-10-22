"""Budget control for shadow replays.

A shadow replay spends real money - every request is answered by a real
candidate model. The budget is therefore a hard stop: the replay halts when
the dollar limit, the request limit or the concurrency limit is reached, and
the partial result is returned with ``truncated=True`` so the caller knows the
evidence is incomplete.

The budget uses a *reserve / settle* model. Before a request is sent the
expected cost is reserved, so concurrent calls can never collectively exceed
the limit; once the call returns the reservation is settled with the real
cost, and the difference returns to the pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(slots=True)
class ReplayBudget:
    """A hard spending limit for a shadow replay.

    Args:
        max_cost_usd: Maximum dollar spend of the replay.
        max_requests: Maximum number of replayed requests.
        max_concurrency: Maximum number of in-flight candidate calls.
    """

    max_cost_usd: float = 1.0
    max_requests: int = 100
    max_concurrency: int = 4
    cost_usd: float = field(default=0.0, init=False)
    requests: int = field(default=0, init=False)
    reserved_usd: float = field(default=0.0, init=False)
    reserved_requests: int = field(default=0, init=False)
    peak_concurrency: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if self.max_cost_usd < 0.0:
            raise ValueError(f"max_cost_usd must be non-negative, got {self.max_cost_usd}")
        if self.max_requests < 0:
            raise ValueError(f"max_requests must be non-negative, got {self.max_requests}")
        if self.max_concurrency <= 0:
            raise ValueError(f"max_concurrency must be positive, got {self.max_concurrency}")

    @property
    def committed_usd(self) -> float:
        """Dollar budget already spent plus reserved for in-flight calls."""
        return self.cost_usd + self.reserved_usd

    @property
    def remaining_cost_usd(self) -> float:
        """Dollar budget still available."""
        return max(0.0, self.max_cost_usd - self.committed_usd)

    @property
    def remaining_requests(self) -> int:
        """Number of requests still affordable."""
        return max(0, self.max_requests - self.requests - self.reserved_requests)

    @property
    def exhausted(self) -> bool:
        """Whether the budget has been fully consumed."""
        return self.remaining_requests == 0 or self.remaining_cost_usd <= 0.0

    def estimate_cost(
        self, requests: Sequence[object], *, cost_per_request_usd: float = 0.0
    ) -> float:
        """Estimate the cost of replaying ``requests``.

        Each item may be a :class:`~shadoweval.capture.ProductionRequest` (its
        ``cost_usd`` is used) or a plain number (interpreted as dollars). When
        ``cost_per_request_usd`` is given it overrides the per-item costs.
        """
        total = 0.0
        for item in requests:
            if cost_per_request_usd:
                total += cost_per_request_usd
                continue
            value = getattr(item, "cost_usd", item)
            try:
                total += float(value)
            except (TypeError, ValueError):  # pragma: no cover - defensive
                continue
        return total

    def can_afford(self, n: int = 1, *, unit_cost_usd: float = 0.0) -> bool:
        """Whether ``n`` more requests fit inside the budget.

        ``unit_cost_usd`` is the expected cost of a single request; when it is
        given the dollar limit is checked as well.
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n > self.remaining_requests:
            return False
        if unit_cost_usd > 0.0:
            return unit_cost_usd * n <= self.remaining_cost_usd
        return self.remaining_cost_usd > 0.0

    def reserve(self, cost_usd: float = 0.0, *, requests: int = 1) -> bool:
        """Reserve ``cost_usd`` for ``requests`` in-flight calls.

        Returns ``False`` (and reserves nothing) when the reservation would
        exceed the budget.
        """
        if requests < 0:
            raise ValueError(f"requests must be non-negative, got {requests}")
        if cost_usd < 0.0:
            raise ValueError(f"cost_usd must be non-negative, got {cost_usd}")
        if requests > self.remaining_requests:
            return False
        if cost_usd > self.remaining_cost_usd:
            return False
        self.reserved_usd += cost_usd
        self.reserved_requests += requests
        self.peak_concurrency = max(self.peak_concurrency, self.reserved_requests)
        return True

    def settle(self, reserved_usd: float, actual_usd: float, *, requests: int = 1) -> None:
        """Settle a reservation with the real cost of the call.

        The difference between the reservation and the real cost returns to
        the available budget.
        """
        if requests < 0:
            raise ValueError(f"requests must be non-negative, got {requests}")
        if actual_usd < 0.0:
            raise ValueError(f"actual_usd must be non-negative, got {actual_usd}")
        self.reserved_usd = max(0.0, self.reserved_usd - reserved_usd)
        self.reserved_requests = max(0, self.reserved_requests - requests)
        self.cost_usd += actual_usd
        self.requests += requests

    def record(self, cost_usd: float = 0.0, *, requests: int = 1) -> None:
        """Record that ``requests`` requests were replayed for ``cost_usd``."""
        if requests < 0:
            raise ValueError(f"requests must be non-negative, got {requests}")
        if cost_usd < 0.0:
            raise ValueError(f"cost_usd must be non-negative, got {cost_usd}")
        self.cost_usd += cost_usd
        self.requests += requests
        self.peak_concurrency = max(self.peak_concurrency, min(requests, self.max_concurrency))

    def acquire(self) -> bool:
        """Reserve one request slot; returns ``False`` when the budget is full."""
        return self.reserve(0.0, requests=1)

    def release(self) -> None:
        """Release one reserved request slot (used when a call is put back)."""
        if self.reserved_requests > 0:
            self.reserved_requests -= 1
        elif self.requests > 0:
            self.requests -= 1

    def summary(self) -> str:
        """A one-line human-readable summary of the budget state."""
        return (
            f"${self.cost_usd:.4f}/${self.max_cost_usd:.2f} spent, "
            f"{self.requests}/{self.max_requests} requests, "
            f"concurrency<={self.max_concurrency}"
        )
