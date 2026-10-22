"""End-to-end tests: capture -> sample -> replay -> decide."""

from __future__ import annotations

import pytest
from conftest import make_request

import shadoweval
from shadoweval import (
    Candidate,
    Decision,
    DivergenceDetector,
    PairedTTest,
    PassthroughResponseGenerator,
    ProductionRequest,
    RegressionAnalyzer,
    ReplayBudget,
    Replayer,
    SamplingStrategy,
    TemplateResponseGenerator,
    TokenOverlapScorer,
    TrafficStore,
)


def test_public_api_is_sorted_and_versioned() -> None:
    assert shadoweval.__version__ == "0.1.0"
    exported = shadoweval.__all__
    assert len(exported) == len(set(exported))
    # ruff RUF022 ordering: uppercase names, then __version__, then lowercase.
    upper = [name for name in exported if name[:1].isupper()]
    lower = [name for name in exported if name[:1].islower()]
    assert exported == [*sorted(upper), "__version__", *sorted(lower)]
    for name in exported:
        assert hasattr(shadoweval, name), name


def test_end_to_end_shadow_evaluation() -> None:
    traffic = TrafficStore()
    for index in range(40):
        traffic.record(
            make_request(
                index,
                tenant=("acme", "globex")[index % 2],
                feedback=1.0 if index % 10 == 0 else None,
                tags=("error",) if index % 13 == 0 else (),
            )
        )

    sample = traffic.sample(30, seed=7, strategy=SamplingStrategy.STRATIFIED)
    assert len(sample) == 30

    candidate = Candidate(
        name="concise-v2",
        prompt_template="Answer briefly.\n{prompt}",
        version="2",
    )
    replayer = Replayer(
        TokenOverlapScorer(),
        generator=TemplateResponseGenerator(quality=0.45),
        detector=DivergenceDetector(threshold=0.6),
    )
    run = replayer.replay(sample, candidate)
    assert len(run) == 30

    verdict = RegressionAnalyzer(min_samples=20).analyze(run.results)
    assert verdict.decision is Decision.BLOCK
    assert verdict.p_value < 0.05
    assert verdict.mean_delta < 0.0
    assert verdict.sample_size == 30
    assert verdict.reasons


def test_end_to_end_equal_candidate_ships() -> None:
    traffic = TrafficStore()
    for index in range(30):
        traffic.record(make_request(index))
    sample = traffic.sample(30, seed=3)
    candidate = Candidate(name="same", prompt_template="{prompt}")
    replayer = Replayer(TokenOverlapScorer(), generator=PassthroughResponseGenerator())
    run = replayer.replay(sample, candidate)
    verdict = RegressionAnalyzer(min_samples=20).analyze(run.results)
    assert verdict.decision is Decision.SHIP
    assert verdict.p_value > 0.05
    assert verdict.mean_delta == pytest.approx(0.0)


def test_end_to_end_budget_limited_run_truncates() -> None:
    traffic = TrafficStore()
    for index in range(50):
        traffic.record(make_request(index))
    sample = traffic.sample(50, seed=3)
    candidate = Candidate(name="same", prompt_template="{prompt}")
    replayer = Replayer(TokenOverlapScorer(), candidate_cost_per_1k_tokens=0.05)
    budget = ReplayBudget(max_cost_usd=0.01, max_requests=50)
    run = replayer.replay(sample, candidate, budget=budget)
    assert run.truncated is True
    assert 0 < len(run) < 50


def test_verdict_is_deterministic_across_runs() -> None:
    traffic = TrafficStore()
    for index in range(30):
        traffic.record(make_request(index))
    sample = traffic.sample(30, seed=13)
    candidate = Candidate(name="same", prompt_template="{prompt}")

    def run_once() -> tuple[float, float]:
        replayer = Replayer(TokenOverlapScorer())
        result = replayer.replay(sample, candidate)
        verdict = RegressionAnalyzer(min_samples=20).analyze(result.results)
        return verdict.p_value, verdict.mean_delta

    assert run_once() == run_once()


def test_sampling_with_error_focused_strategy_surfaces_failures() -> None:
    traffic = TrafficStore()
    for index in range(30):
        traffic.record(make_request(index))
    for index in range(30, 35):
        traffic.record(make_request(index, feedback=1.0))
    sample = traffic.sample(5, seed=2, strategy=SamplingStrategy.ERROR_FOCUSED)
    assert len(sample) == 5
    assert all(request.is_error for request in sample)


def test_divergence_detector_distinguishes_reworded_from_different() -> None:
    detector = DivergenceDetector(threshold=0.6)
    original = "Your refund was processed and will arrive within five business days."
    reworded = "The refund was processed and will reach you within five business days."
    different = "Your subscription was cancelled and no refund was issued."
    assert detector.similarity(original, reworded) >= 0.6
    assert detector.similarity(original, different) < 0.6


def test_paired_t_test_powers_a_known_effect() -> None:
    baseline = [0.70 + 0.001 * index for index in range(40)]
    candidate = [value - 0.05 for value in baseline]
    test = PairedTTest().run(baseline, candidate)
    assert test.p_value < 0.01
    assert test.mean_difference == pytest.approx(-0.05)


def test_production_request_round_trips_through_the_store() -> None:
    request = ProductionRequest(
        request_id="r-1",
        timestamp=make_request(0).timestamp,
        prompt="hello",
        model="m",
        response="world",
        latency_ms=100.0,
        input_tokens=5,
        output_tokens=5,
        cost_usd=0.001,
    )
    traffic = TrafficStore()
    traffic.record(request)
    assert traffic.get("r-1") is request
    assert list(traffic) == [request]


def test_a_degraded_candidate_scores_lower_on_every_request() -> None:
    traffic = TrafficStore()
    for index in range(20):
        traffic.record(make_request(index, response="a full detailed support answer"))
    sample = traffic.sample(20, seed=1)
    candidate = Candidate(name="terse", prompt_template="{prompt}")

    degraded = Replayer(
        TokenOverlapScorer(),
        generator=TemplateResponseGenerator(quality=0.2, template="one word answer"),
    ).replay(sample, candidate)
    healthy = Replayer(TokenOverlapScorer(), generator=PassthroughResponseGenerator()).replay(
        sample, candidate
    )

    assert all(
        degraded_result.candidate_score < healthy_result.candidate_score
        for degraded_result, healthy_result in zip(degraded, healthy, strict=True)
    )
    verdict = RegressionAnalyzer(min_samples=10).analyze(degraded.results)
    assert verdict.decision is Decision.BLOCK


def test_a_candidate_with_a_lower_threshold_flags_more_divergence() -> None:
    request = make_request(0, response="Your invoice is available under Settings and Billing.")
    loose = DivergenceDetector(threshold=0.3)
    strict = DivergenceDetector(threshold=0.95)
    reworded = "The invoice is available under Settings and the Billing page."
    assert loose.diverged(request.response, reworded) is False
    assert strict.diverged(request.response, reworded) is True
