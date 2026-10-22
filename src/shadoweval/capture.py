"""Capture and sampling of production traffic.

The store keeps the *real* requests a service received: prompts, the answers
the production model gave, latency, token counts, cost and whatever human
feedback came back. Sampling decides which of those requests make up the
evaluation set, because replaying all of production traffic is expensive and
because a uniform sample hides regressions that only affect a small slice of
tenants.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np

from shadoweval.text import jaccard_similarity, tokenize

DEFAULT_ERROR_TAGS: frozenset[str] = frozenset(
    {"error", "failure", "failed", "timeout", "exception", "rate_limit", "refusal"}
)


class SamplingStrategy(str, Enum):
    """Ways of choosing an evaluation subset out of a traffic store."""

    UNIFORM = "uniform"
    """Random sample across the whole store."""

    RECENT = "recent"
    """The most recent requests - closest to today's production behaviour."""

    STRATIFIED = "stratified"
    """Round-robin over strata (tenants or tags) so no slice is starved."""

    ERROR_FOCUSED = "error_focused"
    """Prefer requests that errored or collected bad feedback."""


@dataclass(slots=True)
class ProductionRequest:
    """One request captured from production traffic."""

    request_id: str
    timestamp: datetime
    prompt: str
    model: str
    response: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    tenant_id: str = "default"
    tags: tuple[str, ...] = ()
    user_feedback: float | None = None

    @property
    def is_error(self) -> bool:
        """Whether the request looks like a failure worth replaying."""
        return is_error_request(self)

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return int(self.input_tokens) + int(self.output_tokens)


def is_error_request(request: ProductionRequest, *, bad_feedback_threshold: float = 3.0) -> bool:
    """Whether ``request`` represents an error or a badly rated interaction.

    A request counts as an error when it carries one of
    :data:`DEFAULT_ERROR_TAGS`, when its response is empty or starts with a
    refusal marker, or when the user rated it at or below
    ``bad_feedback_threshold`` (the rating scale is assumed to be 1-5).
    """
    if any(tag.lower() in DEFAULT_ERROR_TAGS for tag in request.tags):
        return True
    feedback = request.user_feedback
    if feedback is not None and feedback <= bad_feedback_threshold:
        return True
    response = request.response.strip().lower()
    if not response:
        return True
    return response.startswith(("i cannot", "i can't", "i'm sorry", "i apologize", "error"))


@dataclass(slots=True)
class TrafficStore:
    """An append-only store of production requests."""

    _requests: list[ProductionRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not hasattr(self, "_requests") or self._requests is None:  # pragma: no cover
            self._requests = []

    def record(self, request: ProductionRequest) -> ProductionRequest:
        """Append ``request`` to the store and return it."""
        self._requests.append(request)
        return request

    def extend(self, requests: Iterable[ProductionRequest]) -> None:
        """Append every request in ``requests`` to the store."""
        for request in requests:
            self.record(request)

    def __len__(self) -> int:
        """Number of recorded requests."""
        return len(self._requests)

    def __iter__(self):
        """Iterate over the recorded requests in insertion order."""
        return iter(self._requests)

    def all(self) -> list[ProductionRequest]:
        """A copy of every recorded request, in insertion order."""
        return list(self._requests)

    def get(self, request_id: str) -> ProductionRequest | None:
        """Return the request with ``request_id`` if it was recorded."""
        for request in self._requests:
            if request.request_id == request_id:
                return request
        return None

    def filter_by(
        self,
        *,
        tenant_id: str | None = None,
        model: str | None = None,
        tags: Sequence[str] | None = None,
        min_feedback: float | None = None,
        max_feedback: float | None = None,
        has_feedback: bool | None = None,
        has_error: bool | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        prompt_contains: str | None = None,
    ) -> TrafficStore:
        """Return a new store holding only the requests matching every criterion.

        All criteria are optional and combine with a logical AND. ``tags``
        matches requests carrying *all* of the given tags.
        """
        wanted = {tag.lower() for tag in tags} if tags else None
        selected: list[ProductionRequest] = []
        for request in self._requests:
            if tenant_id is not None and request.tenant_id != tenant_id:
                continue
            if model is not None and request.model != model:
                continue
            if wanted is not None:
                have = {tag.lower() for tag in request.tags}
                if not wanted.issubset(have):
                    continue
            if min_feedback is not None and (
                request.user_feedback is None or request.user_feedback < min_feedback
            ):
                continue
            if max_feedback is not None and (
                request.user_feedback is None or request.user_feedback > max_feedback
            ):
                continue
            if has_feedback is not None and (request.user_feedback is not None) != has_feedback:
                continue
            if has_error is not None and request.is_error != has_error:
                continue
            if since is not None and request.timestamp < since:
                continue
            if until is not None and request.timestamp > until:
                continue
            if (
                prompt_contains is not None
                and prompt_contains.lower() not in request.prompt.lower()
            ):
                continue
            selected.append(request)
        return TrafficStore(selected)

    def sample(
        self,
        n: int,
        *,
        seed: int = 0,
        strategy: SamplingStrategy | str = SamplingStrategy.UNIFORM,
        strata_by: str = "tenant_id",
        dedupe_threshold: float | None = None,
    ) -> list[ProductionRequest]:
        """Draw ``n`` requests using ``strategy``.

        The result is deterministic: the same ``seed``, store and strategy
        always produce the same subset. ``n`` larger than the store returns the
        whole store.

        Args:
            n: How many requests to draw.
            seed: Seed for the internal ``numpy`` generator.
            strategy: A :class:`SamplingStrategy` (or its string value).
            strata_by: Attribute used to build strata for
                :attr:`SamplingStrategy.STRATIFIED` (``tenant_id`` or a tag).
            dedupe_threshold: When given, near-duplicate prompts are removed
                before sampling (see :func:`dedupe`).
        """
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        chosen_strategy = SamplingStrategy(strategy)
        pool = self._requests
        if dedupe_threshold is not None:
            pool = dedupe(pool, similarity_threshold=dedupe_threshold)
        if n >= len(pool):
            return list(pool)
        if n == 0:
            return []
        rng = np.random.default_rng(seed)
        if chosen_strategy is SamplingStrategy.RECENT:
            ordered = sorted(
                enumerate(pool), key=lambda pair: (-pair[1].timestamp.timestamp(), pair[0])
            )
            return [request for _, request in ordered[:n]]
        if chosen_strategy is SamplingStrategy.STRATIFIED:
            return _stratified_sample(pool, n, rng, strata_by)
        if chosen_strategy is SamplingStrategy.ERROR_FOCUSED:
            return _error_focused_sample(pool, n, rng)
        return _uniform_sample(pool, n, rng)


def _uniform_sample(
    pool: Sequence[ProductionRequest], n: int, rng: np.random.Generator
) -> list[ProductionRequest]:
    """Draw ``n`` distinct requests uniformly at random."""
    indices = rng.choice(len(pool), size=n, replace=False)
    return [pool[int(index)] for index in indices]


def _strata_key(request: ProductionRequest, strata_by: str) -> str:
    """Bucket a request into a stratum."""
    if strata_by == "tenant_id":
        return request.tenant_id
    if strata_by == "model":
        return request.model
    if strata_by.startswith("tag:"):
        wanted = strata_by.split(":", 1)[1].lower()
        for tag in request.tags:
            if tag.lower() == wanted:
                return tag
        return f"<no-tag:{wanted}>"
    if strata_by.startswith("feedback:"):
        wanted = strata_by.split(":", 1)[1]
        if request.user_feedback is None:
            return f"<no-{wanted}>"
        return wanted
    return str(getattr(request, strata_by, "<unknown>"))


def _stratified_sample(
    pool: Sequence[ProductionRequest],
    n: int,
    rng: np.random.Generator,
    strata_by: str,
) -> list[ProductionRequest]:
    """Draw ``n`` requests round-robin across strata.

    Each stratum is shuffled independently, then the sampler takes one request
    per stratum in turn, so strata of equal size contribute equally.
    """
    strata: dict[str, list[ProductionRequest]] = {}
    for request in pool:
        strata.setdefault(_strata_key(request, strata_by), []).append(request)
    for key in strata:
        order = rng.permutation(len(strata[key]))
        strata[key] = [strata[key][int(index)] for index in order]

    keys = sorted(strata)
    picked: list[ProductionRequest] = []
    depth = 0
    while len(picked) < n:
        progressed = False
        for key in keys:
            if len(picked) >= n:
                break
            bucket = strata[key]
            if depth < len(bucket):
                picked.append(bucket[depth])
                progressed = True
        if not progressed:
            break
        depth += 1
    return picked


def _error_focused_sample(
    pool: Sequence[ProductionRequest], n: int, rng: np.random.Generator
) -> list[ProductionRequest]:
    """Draw ``n`` requests, preferring the ones that errored or were rated badly."""
    errors = [request for request in pool if request.is_error]
    others = [request for request in pool if not request.is_error]
    picked: list[ProductionRequest] = []
    for bucket in (errors, others):
        if len(picked) >= n:
            break
        take = min(n - len(picked), len(bucket))
        if take <= 0:
            continue
        indices = rng.choice(len(bucket), size=take, replace=False)
        picked.extend(bucket[int(index)] for index in indices)
    return picked


def dedupe(
    requests: Sequence[ProductionRequest], *, similarity_threshold: float = 0.7
) -> list[ProductionRequest]:
    """Drop near-duplicate prompts, keeping the first occurrence of each cluster.

    Two requests are near-duplicates when the Jaccard index of their token sets
    is at least ``similarity_threshold``. This keeps the evaluation set
    diverse: replaying fifty phrasings of the same question fifty times tells
    you nothing about a prompt change.

    The comparison uses the *full* token set (stopwords included) because
    stopwords carry the phrasing of a question - "how do I reset my password"
    and "how do I reset my password again" differ only in a stopword, and
    dropping it would hide the duplication.
    """
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
    kept: list[ProductionRequest] = []
    kept_tokens: list[set[str]] = []
    for request in requests:
        tokens = set(tokenize(request.prompt))
        duplicate = False
        for existing in kept_tokens:
            if jaccard_similarity(tokens, existing) >= similarity_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(request)
        kept_tokens.append(tokens)
    return kept


def dedupe_prompts(prompts: Sequence[str], *, similarity_threshold: float = 0.7) -> list[str]:
    """Drop near-duplicate prompts, keeping the first occurrence of each cluster."""
    if not 0.0 <= similarity_threshold <= 1.0:
        raise ValueError(f"similarity_threshold must be in [0, 1], got {similarity_threshold}")
    kept: list[str] = []
    kept_tokens: list[set[str]] = []
    for prompt in prompts:
        tokens = set(tokenize(prompt))
        if any(
            jaccard_similarity(tokens, existing) >= similarity_threshold for existing in kept_tokens
        ):
            continue
        kept.append(prompt)
        kept_tokens.append(tokens)
    return kept


def prompt_fingerprint(prompt: str) -> str:
    """A stable short fingerprint of ``prompt`` (used for cheap exact dedupe)."""
    return hashlib.blake2b(prompt.encode("utf-8"), digest_size=8).hexdigest()
