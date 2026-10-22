"""Scoring of model responses, plus the distribution statistics used to judge them.

Every scorer turns a (request, response) pair into a number in ``[0, 1]``.
The same scorer is applied to the production response and to the candidate
response, so the *difference* of the two scores is a fair, paired observation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from shadoweval.capture import ProductionRequest
from shadoweval.text import content_tokens, tokenize


@runtime_checkable
class Scorer(Protocol):
    """Contract every scorer implements."""

    def score(self, request: ProductionRequest, response: str) -> float:
        """Score ``response`` (given ``request``) on a ``[0, 1]`` scale."""
        ...


def _clamp(value: float) -> float:
    """Clamp ``value`` into the ``[0, 1]`` range."""
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else float(value)


@dataclass(slots=True)
class ExactMatchScorer:
    """Scores 1.0 when the response matches the reference answer exactly.

    The comparison is case- and whitespace-insensitive; ``expected`` may be
    supplied to compare against a golden answer instead of the recorded
    production response.
    """

    expected: str | None = None
    normalize: bool = True

    def score(self, request: ProductionRequest, response: str) -> float:
        """Return 1.0 for an exact match, 0.0 otherwise."""
        reference = self.expected if self.expected is not None else request.response
        if self.normalize:
            return (
                1.0
                if " ".join(response.split()).lower() == " ".join(reference.split()).lower()
                else 0.0
            )
        return 1.0 if response == reference else 0.0


@dataclass(slots=True)
class TokenOverlapScorer:
    """Token-level F1 between the response and the reference.

    F1 is the harmonic mean of precision (how much of the response is correct)
    and recall (how much of the reference was covered). It is the standard
    metric for open-ended generation tasks and is fully deterministic.
    """

    expected: str | None = None
    use_content_tokens: bool = True

    def score(self, request: ProductionRequest, response: str) -> float:
        """Return the token-level F1 of ``response`` against the reference."""
        reference = self.expected if self.expected is not None else request.response
        if self.use_content_tokens:
            predicted_tokens = content_tokens(response)
            reference_tokens = content_tokens(reference)
        else:
            predicted_tokens = tokenize(response)
            reference_tokens = tokenize(reference)
        return token_f1(predicted_tokens, reference_tokens)


def token_f1(predicted: Sequence[str], reference: Sequence[str]) -> float:
    """Token-level F1 between two token sequences.

    Multiset overlap is used, so a repeated word counts as many times as it
    appears in both sequences. Two empty sequences score ``1.0``.
    """
    if not predicted and not reference:
        return 1.0
    if not predicted or not reference:
        return 0.0
    predicted_counts: dict[str, int] = {}
    for token in predicted:
        predicted_counts[token] = predicted_counts.get(token, 0) + 1
    reference_counts: dict[str, int] = {}
    for token in reference:
        reference_counts[token] = reference_counts.get(token, 0) + 1
    overlap = 0
    for token, count in predicted_counts.items():
        overlap += min(count, reference_counts.get(token, 0))
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2.0 * precision * recall / (precision + recall)


@dataclass(slots=True)
class RubricScorer:
    """Keyword-coverage scorer: the fraction of rubric keywords present.

    Each keyword counts as present when it appears in the response, either as a
    whole token or as a substring of a token (so ``"refund"`` matches
    ``"refunds"``). The score is the share of rubric keywords covered.
    """

    keywords: tuple[str, ...] = ()
    expected: str | None = None

    def score(self, request: ProductionRequest, response: str) -> float:
        """Return the fraction of rubric keywords covered by ``response``."""
        reference = self.expected if self.expected is not None else request.response
        # Without an explicit rubric, fall back to keyword coverage of the
        # reference answer's content tokens.
        keywords = self.keywords or tuple(content_tokens(reference))
        if not keywords:
            return 0.0
        response_lower = response.lower()
        response_tokens = set(tokenize(response))
        hits = 0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            if keyword_lower in response_tokens or keyword_lower in response_lower:
                hits += 1
        return hits / len(keywords)


@dataclass(slots=True)
class LengthPenaltyScorer:
    """Penalises responses that are far too short or far too long.

    The score is ``1.0`` when the candidate length is within
    ``tolerance`` of the reference length ratio, and decays smoothly towards
    ``0.0`` as the ratio drifts away.
    """

    tolerance: float = 0.25
    min_length: int = 1

    def score(self, request: ProductionRequest, response: str) -> float:
        """Return a length-similarity score in ``[0, 1]``."""
        reference_length = len(tokenize(request.response))
        candidate_length = len(tokenize(response))
        if reference_length < self.min_length and candidate_length < self.min_length:
            return 1.0
        ratio = candidate_length / max(reference_length, 1)
        deviation = abs(1.0 - ratio)
        if deviation <= self.tolerance:
            return 1.0
        span = max(1.0 - self.tolerance, 1e-9)
        return _clamp(1.0 - (deviation - self.tolerance) / span)


@dataclass(slots=True)
class CompositeScorer:
    """Weighted combination of several scorers.

    The weights are normalised so they need not sum to one. Sub-scorers are
    combined with a weighted average.
    """

    scorers: tuple[Scorer, ...] = ()
    weights: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.weights and len(self.weights) != len(self.scorers):
            raise ValueError(
                f"weights length {len(self.weights)} does not match {len(self.scorers)} scorers"
            )
        if not self.scorers:
            raise ValueError("CompositeScorer needs at least one sub-scorer")
        if not self.weights:
            object.__setattr__(self, "weights", tuple(1.0 for _ in self.scorers))
        total = sum(abs(weight) for weight in self.weights)
        if total <= 0.0:
            raise ValueError("at least one weight must be non-zero")
        object.__setattr__(self, "weights", tuple(weight / total for weight in self.weights))

    def score(self, request: ProductionRequest, response: str) -> float:
        """Return the weighted average of the sub-scores."""
        total = 0.0
        for scorer, weight in zip(self.scorers, self.weights, strict=True):
            total += weight * _clamp(scorer.score(request, response))
        return _clamp(total)


@dataclass(slots=True)
class ConfidenceInterval:
    """A two-sided confidence interval around a mean."""

    lower: float
    upper: float
    confidence: float = 0.95

    @property
    def width(self) -> float:
        """Width of the interval (``upper - lower``)."""
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        """Whether ``value`` lies inside the interval."""
        return self.lower <= value <= self.upper


@dataclass(slots=True)
class ScoreDistribution:
    """Summary statistics of a list of scores, with a bootstrap CI for the mean."""

    scores: tuple[float, ...] = ()
    mean: float = 0.0
    std: float = 0.0
    minimum: float = 0.0
    maximum: float = 0.0
    percentiles: dict[int, float] = field(default_factory=dict)
    confidence_interval: ConfidenceInterval | None = None
    n: int = 0

    @classmethod
    def from_scores(
        cls,
        scores: Sequence[float],
        *,
        confidence: float = 0.95,
        resamples: int = 1000,
        seed: int = 0,
    ) -> ScoreDistribution:
        """Build a distribution from ``scores``.

        The confidence interval of the mean is estimated by a seeded bootstrap:
        ``resamples`` resamples of the score list are drawn with replacement and
        the 2.5/97.5 percentiles of their means form the interval.
        """
        values = np.asarray(list(scores), dtype=np.float64)
        count = int(values.size)
        if count == 0:
            return cls(scores=(), mean=0.0, std=0.0, minimum=0.0, maximum=0.0, percentiles={})
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if count > 1 else 0.0
        percentiles = {int(p): float(np.percentile(values, p)) for p in (10, 25, 50, 75, 90, 95)}
        interval = bootstrap_mean_ci(values, confidence=confidence, resamples=resamples, seed=seed)
        return cls(
            scores=tuple(float(value) for value in values),
            mean=mean,
            std=std,
            minimum=float(values.min()),
            maximum=float(values.max()),
            percentiles=percentiles,
            confidence_interval=interval,
            n=count,
        )

    def percentile(self, percentile: float) -> float:
        """Return the ``percentile``-th percentile of the scores."""
        if not self.scores:
            raise ValueError("cannot take a percentile of an empty distribution")
        return float(np.percentile(np.asarray(self.scores, dtype=np.float64), percentile))


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    *,
    confidence: float = 0.95,
    resamples: int = 1000,
    seed: int = 0,
) -> ConfidenceInterval:
    """Bootstrap confidence interval for the mean of ``values``.

    The estimator is deterministic: the same ``seed`` always yields the same
    interval.
    """
    array = np.asarray(values, dtype=np.float64)
    count = int(array.size)
    if count == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if resamples <= 0:
        raise ValueError(f"resamples must be positive, got {resamples}")
    if count == 1:
        value = float(array[0])
        return ConfidenceInterval(lower=value, upper=value, confidence=confidence)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, count, size=(resamples, count))
    means = array[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    lower = float(np.percentile(means, 100.0 * tail))
    upper = float(np.percentile(means, 100.0 * (1.0 - tail)))
    return ConfidenceInterval(lower=lower, upper=upper, confidence=confidence)
