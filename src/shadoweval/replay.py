"""Replaying production traffic against a candidate prompt or model.

A *replay* takes the requests production already answered, asks the candidate
for an answer to the same prompts, and scores both answers with the same
scorer. The resulting per-request differences are the paired observations the
decision layer feeds into a t-test.

Everything here is deterministic: the candidate is either a real provider
adapter (implement :class:`ResponseGenerator`) or, as in the bundled example,
a deterministic local stand-in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from shadoweval.budget import ReplayBudget
from shadoweval.capture import ProductionRequest
from shadoweval.scoring import Scorer, TokenOverlapScorer
from shadoweval.text import content_tokens, extract_numbers, jaccard_similarity, tokenize


@dataclass(slots=True)
class Candidate:
    """A candidate prompt or model to evaluate against production traffic."""

    name: str
    prompt_template: str | None = None
    model: str | None = None
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("candidate name must not be empty")
        if self.prompt_template is None and self.model is None:
            raise ValueError("a candidate needs either a prompt_template or a model")
        if self.prompt_template is None and self.model is not None:
            self.prompt_template = "{prompt}"

    def render(self, request: ProductionRequest) -> str:
        """Render the candidate's prompt template for ``request``."""
        template = self.prompt_template or "{prompt}"
        return template.format(prompt=request.prompt, model=request.model)


@dataclass(slots=True)
class ReplayResult:
    """The outcome of replaying one production request against a candidate."""

    request_id: str
    baseline_response: str
    candidate_response: str
    baseline_score: float
    candidate_score: float
    latency_delta_ms: float
    cost_delta_usd: float
    diverged: bool
    similarity: float = 1.0
    baseline_cost_usd: float = 0.0
    candidate_cost_usd: float = 0.0
    baseline_latency_ms: float = 0.0
    candidate_latency_ms: float = 0.0
    baseline_tokens: int = 0
    candidate_tokens: int = 0

    @property
    def score_delta(self) -> float:
        """Difference between the candidate and baseline scores."""
        return self.candidate_score - self.baseline_score


@runtime_checkable
class ResponseGenerator(Protocol):
    """Produces a response for a rendered candidate prompt."""

    def generate(self, prompt: str, request: ProductionRequest) -> str:
        """Return the candidate's answer to ``prompt``."""
        ...


@dataclass(slots=True)
class TemplateResponseGenerator:
    """A deterministic, offline stand-in for a model.

    The generator is a pure function of the prompt: it extracts the salient
    terms of the prompt and returns a templated answer. Degrading or improving
    the answer is done by passing ``quality`` (a multiplier on the number of
    salient terms covered) and ``extra_padding`` (tokens appended to make the
    candidate wordier and therefore more expensive).
    """

    quality: float = 1.0
    extra_padding: int = 0
    template: str = "{summary}"

    def generate(self, prompt: str, request: ProductionRequest) -> str:
        """Return a deterministic answer for ``prompt``."""
        terms = list(dict.fromkeys(content_tokens(prompt)))
        if not terms:
            terms = list(dict.fromkeys(tokenize(prompt))) or ["unknown"]
        keep = max(1, round(len(terms) * max(0.0, min(1.0, self.quality))))
        summary = " ".join(terms[:keep])
        answer = self.template.format(summary=summary)
        if self.extra_padding:
            answer = f"{answer} {' '.join(['detail'] * self.extra_padding)}"
        return answer


@dataclass(slots=True)
class PassthroughResponseGenerator:
    """A candidate that reproduces the production answer verbatim.

    This is the "no-op candidate": it models a redeploy of the same model, or
    a prompt change that provably cannot alter the answer. Replaying it is
    the sanity check of a shadow harness - every score must match the
    baseline exactly.
    """

    def generate(self, prompt: str, request: ProductionRequest) -> str:
        """Return the recorded production answer."""
        return request.response


@dataclass(slots=True)
class DivergenceDetector:
    """Decides whether two responses are semantically equivalent.

    The score combines two signals:

    1. **Normalised token overlap** - the Jaccard index of the content tokens
       of both answers, so a reworded-but-equivalent answer still scores high.
    2. **A numeric / statement check** - when both answers contain numbers they
       must agree, and when one answer negates a statement the other must not.
       A numeric disagreement caps the score, because "the fee is 12" and
       "the fee is 15" share many words but mean opposite things.
    """

    threshold: float = 0.8
    token_weight: float = 0.8
    numeric_weight: float = 0.2
    negation_penalty: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")
        if self.token_weight < 0.0 or self.numeric_weight < 0.0:
            raise ValueError("weights must be non-negative")

    def similarity(self, a: str, b: str) -> float:
        """Return the semantic similarity of ``a`` and ``b`` in ``[0, 1]``."""
        token_sim = jaccard_similarity(content_tokens(a), content_tokens(b))
        numbers_a = extract_numbers(a)
        numbers_b = extract_numbers(b)
        if numbers_a and numbers_b:
            numeric_sim = 1.0 if numbers_a == numbers_b else 0.0
        elif numbers_a or numbers_b:
            # One side carries a number and the other does not: only a partial
            # match, since the numeric claim is unverified.
            numeric_sim = 0.5
        else:
            numeric_sim = 1.0
        score = self.token_weight * token_sim + self.numeric_weight * numeric_sim
        if self._negation_conflict(a, b):
            score -= self.negation_penalty
        return max(0.0, min(1.0, score))

    def _negation_conflict(self, a: str, b: str) -> bool:
        """Whether exactly one of the two answers negates the statement."""
        from shadoweval.text import has_negation

        return has_negation(a) != has_negation(b)

    def diverged(self, a: str, b: str) -> bool:
        """Whether ``a`` and ``b`` diverge beyond the configured threshold."""
        return self.similarity(a, b) < self.threshold

    def detect(self, a: str, b: str) -> tuple[bool, float]:
        """Return ``(diverged, similarity)`` for the pair ``a`` / ``b``."""
        score = self.similarity(a, b)
        return score < self.threshold, score


@dataclass(slots=True)
class ReplayRun:
    """The outcome of a whole replay run."""

    results: list[ReplayResult] = field(default_factory=list)
    truncated: bool = False
    requests_attempted: int = 0
    cost_usd: float = 0.0

    def __len__(self) -> int:
        """Number of replayed requests."""
        return len(self.results)

    def __iter__(self):
        """Iterate over the individual replay results."""
        return iter(self.results)

    @property
    def baseline_scores(self) -> list[float]:
        """Baseline scores of every replayed request."""
        return [result.baseline_score for result in self.results]

    @property
    def candidate_scores(self) -> list[float]:
        """Candidate scores of every replayed request."""
        return [result.candidate_score for result in self.results]

    @property
    def divergence_rate(self) -> float:
        """Share of replayed requests whose answer changed meaningfully."""
        if not self.results:
            return 0.0
        return sum(1 for result in self.results if result.diverged) / len(self.results)


class Replayer:
    """Replays a sample of production traffic against a candidate.

    Args:
        scorer: The scorer applied to both the baseline and the candidate
            answer, so that the difference of the two scores is meaningful.
        generator: Produces candidate answers. Defaults to a deterministic
            :class:`TemplateResponseGenerator`.
        detector: Decides whether two answers diverged.
        candidate_cost_per_1k_tokens: Cost model of the candidate model, used
            to turn token counts into dollars.
        candidate_latency_ms: Latency of the candidate model, in milliseconds.
        cost_per_1k_tokens: Cost model of the production model.
    """

    def __init__(
        self,
        scorer: Scorer | None = None,
        *,
        generator: ResponseGenerator | None = None,
        detector: DivergenceDetector | None = None,
        candidate_cost_per_1k_tokens: float = 0.003,
        candidate_latency_ms: float = 900.0,
        cost_per_1k_tokens: float = 0.003,
    ) -> None:
        self._scorer = scorer or TokenOverlapScorer()
        self._generator = generator or TemplateResponseGenerator()
        self._detector = detector or DivergenceDetector()
        self._candidate_cost_per_1k = candidate_cost_per_1k_tokens
        self._candidate_latency_ms = candidate_latency_ms
        self._cost_per_1k = cost_per_1k_tokens

    @property
    def scorer(self) -> Scorer:
        """The scorer used for both baseline and candidate answers."""
        return self._scorer

    @property
    def detector(self) -> DivergenceDetector:
        """The divergence detector."""
        return self._detector

    def estimate_cost(self, requests: Sequence[ProductionRequest]) -> float:
        """Estimate the dollar cost of replaying ``requests``."""
        total = 0.0
        for request in requests:
            total += self.estimate_request_cost(request)
        return total

    def estimate_request_cost(self, request: ProductionRequest) -> float:
        """Estimate the dollar cost of replaying a single ``request``.

        The *total* production token count (prompt plus completion) is used
        as a proxy for the candidate's token count. A candidate model bills
        for the prompt it re-reads as well as for the answer it writes, so
        this estimate is deliberately conservative: the budget is charged
        before the request is sent, never after.
        """
        return max(1, request.total_tokens) / 1000.0 * self._candidate_cost_per_1k

    def replay(
        self,
        requests: Sequence[ProductionRequest],
        candidate: Candidate,
        *,
        budget: ReplayBudget | None = None,
    ) -> ReplayRun:
        """Replay ``requests`` against ``candidate``.

        When ``budget`` is given the run stops as soon as the budget is
        exhausted; the returned :class:`ReplayRun` then carries
        ``truncated=True`` and holds the partial results.
        """
        run = ReplayRun()
        for request in requests:
            estimate = self.estimate_request_cost(request)
            if budget is not None and not budget.reserve(estimate, requests=1):
                run.truncated = True
                break
            result = self._replay_one(request, candidate)
            run.results.append(result)
            run.requests_attempted += 1
            run.cost_usd += result.candidate_cost_usd
            if budget is not None:
                budget.settle(estimate, result.candidate_cost_usd, requests=1)
        return run

    def _replay_one(self, request: ProductionRequest, candidate: Candidate) -> ReplayResult:
        """Replay a single request."""
        rendered = candidate.render(request)
        candidate_response = self._generator.generate(rendered, request)
        baseline_score = float(self._scorer.score(request, request.response))
        candidate_score = float(self._scorer.score(request, candidate_response))
        diverged, similarity = self._detector.detect(request.response, candidate_response)

        candidate_tokens = max(1, len(tokenize(candidate_response)))
        baseline_tokens = max(1, request.output_tokens)
        # A candidate re-reads the prompt and writes a new completion, so it
        # is billed for the prompt tokens as well.
        candidate_cost = (
            (request.input_tokens + candidate_tokens) / 1000.0 * self._candidate_cost_per_1k
        )
        baseline_cost = (request.input_tokens + baseline_tokens) / 1000.0 * self._cost_per_1k
        candidate_latency = self._candidate_latency_ms

        return ReplayResult(
            request_id=request.request_id,
            baseline_response=request.response,
            candidate_response=candidate_response,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            latency_delta_ms=candidate_latency - request.latency_ms,
            cost_delta_usd=candidate_cost - baseline_cost,
            diverged=diverged,
            similarity=similarity,
            baseline_cost_usd=baseline_cost,
            candidate_cost_usd=candidate_cost,
            baseline_latency_ms=request.latency_ms,
            candidate_latency_ms=candidate_latency,
            baseline_tokens=baseline_tokens,
            candidate_tokens=candidate_tokens,
        )


def replay_traffic(
    requests: Sequence[ProductionRequest],
    candidate: Candidate,
    *,
    scorer: Scorer | None = None,
    generator: ResponseGenerator | None = None,
    detector: DivergenceDetector | None = None,
    budget: ReplayBudget | None = None,
    candidate_cost_per_1k_tokens: float = 0.003,
    candidate_latency_ms: float = 900.0,
) -> ReplayRun:
    """Convenience wrapper that builds a :class:`Replayer` and runs it."""
    replayer = Replayer(
        scorer,
        generator=generator,
        detector=detector,
        candidate_cost_per_1k_tokens=candidate_cost_per_1k_tokens,
        candidate_latency_ms=candidate_latency_ms,
    )
    return replayer.replay(requests, candidate, budget=budget)


__all__ = [
    "Candidate",
    "DivergenceDetector",
    "ReplayResult",
    "ReplayRun",
    "Replayer",
    "ResponseGenerator",
    "TemplateResponseGenerator",
    "replay_traffic",
]
