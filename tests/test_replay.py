"""Tests for the replayer, the divergence detector and the replay budget."""

from __future__ import annotations

import pytest
from conftest import make_request

from shadoweval import (
    Candidate,
    DivergenceDetector,
    ReplayBudget,
    Replayer,
    ReplayResult,
    TemplateResponseGenerator,
    TokenOverlapScorer,
    replay_traffic,
)


def test_candidate_requires_a_template_or_a_model() -> None:
    with pytest.raises(ValueError, match="prompt_template or a model"):
        Candidate(name="broken")


def test_candidate_requires_a_name() -> None:
    with pytest.raises(ValueError, match="name must not be empty"):
        Candidate(name="", prompt_template="{prompt}")


def test_candidate_renders_its_template() -> None:
    request = make_request(0, prompt="reset my password")
    candidate = Candidate(name="c1", prompt_template="Q: {prompt}\nA:")
    assert candidate.render(request) == "Q: reset my password\nA:"


def test_candidate_defaults_the_template_to_the_model() -> None:
    request = make_request(0, prompt="hello")
    candidate = Candidate(name="c1", model="gpt-x")
    assert candidate.render(request) == "hello"


def test_divergence_detector_scores_identical_answers_high() -> None:
    detector = DivergenceDetector()
    answer = "The invoice is available under Settings and Billing."
    assert detector.similarity(answer, answer) == pytest.approx(1.0)
    assert detector.diverged(answer, answer) is False


def test_divergence_detector_scores_a_reworded_answer_high() -> None:
    detector = DivergenceDetector(threshold=0.6)
    original = "Your invoice is available under Settings and Billing."
    reworded = "The invoice can be found under Settings and Billing."
    assert detector.similarity(original, reworded) >= 0.6
    assert detector.diverged(original, reworded) is False


def test_divergence_detector_scores_a_different_answer_low() -> None:
    detector = DivergenceDetector(threshold=0.6)
    original = "Your invoice is available under Settings and Billing."
    different = "To upgrade your plan, contact the sales team for a quote."
    assert detector.similarity(original, different) < 0.6
    assert detector.diverged(original, different) is True


def test_divergence_detector_handles_numeric_answers() -> None:
    detector = DivergenceDetector(threshold=0.6)
    same = "The fee is 12 dollars."
    reworded = "The fee comes to 12 dollars."
    different = "The fee is 15 dollars."
    assert detector.similarity(same, reworded) >= 0.6
    assert detector.similarity(same, different) < 0.6


def test_divergence_detector_penalises_negation_conflicts() -> None:
    detector = DivergenceDetector(threshold=0.6)
    positive = "The account is active."
    negative = "The account is not active."
    assert detector.similarity(positive, negative) < detector.similarity(positive, positive)


def test_divergence_detector_detect_returns_both_values() -> None:
    detector = DivergenceDetector(threshold=0.6)
    diverged, similarity = detector.detect("alpha beta", "gamma delta")
    assert diverged is True
    assert 0.0 <= similarity <= 1.0


def test_divergence_detector_rejects_an_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        DivergenceDetector(threshold=1.5)


def test_divergence_detector_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        DivergenceDetector(token_weight=-1.0)


def test_replay_scores_baseline_and_candidate_identically() -> None:
    request = make_request(0, response="The invoice is under Settings.")
    replayer = Replayer(TokenOverlapScorer())
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    run = replayer.replay([request], candidate)
    assert len(run) == 1
    result = run.results[0]
    assert result.request_id == request.request_id
    # The baseline answer is scored against itself, so it is a perfect 1.0.
    assert result.baseline_score == pytest.approx(1.0)
    assert 0.0 <= result.candidate_score <= 1.0
    assert 0.0 <= result.similarity <= 1.0


def test_replay_marks_an_unchanged_answer_as_not_diverged() -> None:
    request = make_request(0, response="The invoice is under Settings.")
    replayer = Replayer(
        generator=TemplateResponseGenerator(template="The invoice is under Settings."),
        detector=DivergenceDetector(threshold=0.8),
    )
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    result = replayer.replay([request], candidate).results[0]
    assert result.similarity == pytest.approx(1.0)
    assert result.diverged is False


def test_replay_is_deterministic() -> None:
    requests = [make_request(index) for index in range(10)]
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    first = Replayer().replay(requests, candidate)
    second = Replayer().replay(requests, candidate)
    assert [r.candidate_response for r in first] == [r.candidate_response for r in second]
    assert [r.baseline_score for r in first] == [r.baseline_score for r in second]


def test_replay_reports_cost_and_latency_deltas() -> None:
    request = make_request(0)
    replayer = Replayer(candidate_cost_per_1k_tokens=0.01, candidate_latency_ms=1500.0)
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    result = replayer.replay([request], candidate).results[0]
    assert result.candidate_latency_ms == 1500.0
    assert result.latency_delta_ms == pytest.approx(1500.0 - request.latency_ms)
    assert result.candidate_cost_usd > 0.0
    assert result.cost_delta_usd == pytest.approx(
        result.candidate_cost_usd - result.baseline_cost_usd
    )


def test_replay_flags_divergence() -> None:
    request = make_request(0, response="The invoice is under Settings.")
    replayer = Replayer(
        generator=TemplateResponseGenerator(template="Completely different answer."),
        detector=DivergenceDetector(threshold=0.6),
    )
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    result = replayer.replay([request], candidate).results[0]
    assert result.diverged is True


def test_replay_run_reports_the_divergence_rate() -> None:
    requests = [make_request(index) for index in range(4)]
    replayer = Replayer()
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    run = replayer.replay(requests, candidate)
    assert 0.0 <= run.divergence_rate <= 1.0
    assert run.requests_attempted == 4


def test_replay_estimates_the_cost() -> None:
    requests = [make_request(index) for index in range(5)]
    replayer = Replayer(candidate_cost_per_1k_tokens=0.002)
    assert replayer.estimate_cost(requests) > 0.0


def test_replay_traffic_convenience_wrapper() -> None:
    requests = [make_request(index) for index in range(3)]
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    run = replay_traffic(requests, candidate)
    assert len(run) == 3


def test_template_generator_is_deterministic_and_shorter_at_lower_quality() -> None:
    request = make_request(0, prompt="reset my password on the mobile app today")
    good = TemplateResponseGenerator(quality=1.0).generate("reset my password", request)
    poor = TemplateResponseGenerator(quality=0.4).generate("reset my password", request)
    assert good == TemplateResponseGenerator(quality=1.0).generate("reset my password", request)
    assert len(poor.split()) < len(good.split())


def test_template_generator_handles_an_empty_prompt() -> None:
    request = make_request(0, prompt="")
    answer = TemplateResponseGenerator().generate("", request)
    assert answer


# --------------------------------------------------------------------------
# Budget
# --------------------------------------------------------------------------


def test_budget_can_afford_respects_the_request_limit() -> None:
    budget = ReplayBudget(max_cost_usd=10.0, max_requests=5)
    assert budget.can_afford(5) is True
    assert budget.can_afford(6) is False


def test_budget_can_afford_respects_the_cost_limit() -> None:
    budget = ReplayBudget(max_cost_usd=0.05, max_requests=100)
    assert budget.can_afford(10, unit_cost_usd=0.001) is True
    assert budget.can_afford(100, unit_cost_usd=0.001) is False


def test_budget_can_afford_rejects_a_negative_count() -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        ReplayBudget().can_afford(-1)


def test_budget_records_spend() -> None:
    budget = ReplayBudget(max_cost_usd=1.0, max_requests=10)
    budget.record(0.25, requests=1)
    assert budget.cost_usd == pytest.approx(0.25)
    assert budget.requests == 1
    assert budget.remaining_cost_usd == pytest.approx(0.75)
    assert budget.remaining_requests == 9
    assert budget.exhausted is False


def test_budget_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_cost_usd"):
        ReplayBudget(max_cost_usd=-1.0)
    with pytest.raises(ValueError, match="max_requests"):
        ReplayBudget(max_requests=-1)
    with pytest.raises(ValueError, match="max_concurrency"):
        ReplayBudget(max_concurrency=0)


def test_budget_rejects_invalid_records() -> None:
    budget = ReplayBudget()
    with pytest.raises(ValueError, match="cost_usd"):
        budget.record(-0.1)
    with pytest.raises(ValueError, match="requests"):
        budget.record(0.1, requests=-1)


def test_budget_estimates_cost_from_requests() -> None:
    requests = [make_request(index) for index in range(4)]
    budget = ReplayBudget()
    expected = sum(request.cost_usd for request in requests)
    assert budget.estimate_cost(requests) == pytest.approx(expected)


def test_budget_estimates_cost_with_a_per_request_override() -> None:
    requests = [make_request(index) for index in range(4)]
    budget = ReplayBudget()
    assert budget.estimate_cost(requests, cost_per_request_usd=0.01) == pytest.approx(0.04)


def test_budget_acquire_and_release() -> None:
    budget = ReplayBudget(max_cost_usd=1.0, max_requests=1)
    assert budget.acquire() is True
    assert budget.acquire() is False
    budget.release()
    assert budget.acquire() is True


def test_budget_reserve_and_settle() -> None:
    budget = ReplayBudget(max_cost_usd=1.0, max_requests=10)
    assert budget.reserve(0.5, requests=1) is True
    assert budget.reserved_usd == pytest.approx(0.5)
    assert budget.committed_usd == pytest.approx(0.5)
    assert budget.remaining_cost_usd == pytest.approx(0.5)
    budget.settle(0.5, 0.2, requests=1)
    assert budget.reserved_usd == pytest.approx(0.0)
    assert budget.cost_usd == pytest.approx(0.2)
    assert budget.requests == 1
    assert budget.remaining_cost_usd == pytest.approx(0.8)


def test_budget_reserve_fails_when_the_limit_would_be_exceeded() -> None:
    budget = ReplayBudget(max_cost_usd=0.1, max_requests=10)
    assert budget.reserve(0.2, requests=1) is False
    assert budget.reserved_usd == 0.0
    assert budget.reserve(0.1, requests=1) is True
    assert budget.reserve(0.01, requests=1) is False


def test_budget_reserve_fails_on_the_request_limit() -> None:
    budget = ReplayBudget(max_cost_usd=10.0, max_requests=1)
    assert budget.reserve(0.0, requests=1) is True
    assert budget.reserve(0.0, requests=1) is False


def test_budget_reserve_rejects_invalid_values() -> None:
    budget = ReplayBudget()
    with pytest.raises(ValueError, match="cost_usd"):
        budget.reserve(-1.0)
    with pytest.raises(ValueError, match="requests"):
        budget.reserve(0.0, requests=-1)


def test_budget_settle_rejects_invalid_values() -> None:
    budget = ReplayBudget()
    with pytest.raises(ValueError, match="actual_usd"):
        budget.settle(0.1, -1.0)
    with pytest.raises(ValueError, match="requests"):
        budget.settle(0.1, 0.1, requests=-1)


def test_budget_summary_mentions_the_limits() -> None:
    summary = ReplayBudget(max_cost_usd=2.0, max_requests=10).summary()
    assert "$" in summary
    assert "requests" in summary


def test_replay_stops_when_the_budget_is_exhausted() -> None:
    requests = [make_request(index) for index in range(20)]
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    replayer = Replayer(candidate_cost_per_1k_tokens=1.0, candidate_latency_ms=900.0)
    budget = ReplayBudget(max_cost_usd=0.02, max_requests=100)
    run = replayer.replay(requests, candidate, budget=budget)
    assert run.truncated is True
    assert len(run) < len(requests)
    assert budget.cost_usd <= 0.02 + 1e-9


def test_replay_is_not_truncated_when_the_budget_is_generous() -> None:
    requests = [make_request(index) for index in range(10)]
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    budget = ReplayBudget(max_cost_usd=100.0, max_requests=100)
    run = Replayer().replay(requests, candidate, budget=budget)
    assert run.truncated is False
    assert len(run) == 10


def test_replay_truncates_on_the_request_limit() -> None:
    requests = [make_request(index) for index in range(20)]
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    budget = ReplayBudget(max_cost_usd=100.0, max_requests=6)
    run = Replayer().replay(requests, candidate, budget=budget)
    assert run.truncated is True
    assert len(run) == 6
    assert budget.requests == 6


def test_replay_result_exposes_the_score_delta() -> None:
    result = ReplayResult(
        request_id="r",
        baseline_response="a",
        candidate_response="b",
        baseline_score=0.4,
        candidate_score=0.7,
        latency_delta_ms=0.0,
        cost_delta_usd=0.0,
        diverged=True,
    )
    assert result.score_delta == pytest.approx(0.3)


def test_replay_run_exposes_the_score_columns() -> None:
    requests = [make_request(index) for index in range(3)]
    candidate = Candidate(name="c1", prompt_template="{prompt}")
    run = Replayer().replay(requests, candidate)
    assert len(run.baseline_scores) == 3
    assert len(run.candidate_scores) == 3
    assert all(0.0 <= score <= 1.0 for score in run.baseline_scores)


def test_empty_replay_run_has_a_zero_divergence_rate() -> None:
    from shadoweval.replay import ReplayRun

    assert ReplayRun().divergence_rate == 0.0


def test_replayer_exposes_its_scorer_and_detector() -> None:
    scorer = TokenOverlapScorer()
    detector = DivergenceDetector(threshold=0.5)
    replayer = Replayer(scorer, detector=detector)
    assert replayer.scorer is scorer
    assert replayer.detector is detector


def test_template_generator_appends_padding() -> None:
    request = make_request(0, prompt="reset my password")
    padded = TemplateResponseGenerator(extra_padding=3).generate("reset my password", request)
    assert padded.count("detail") == 3


def test_budget_release_falls_back_to_recorded_requests() -> None:
    budget = ReplayBudget(max_cost_usd=1.0, max_requests=5)
    budget.record(0.1, requests=1)
    budget.release()
    assert budget.requests == 0
