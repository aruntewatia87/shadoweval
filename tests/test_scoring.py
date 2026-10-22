"""Tests for the scorers, token F1 and the score distribution statistics."""

from __future__ import annotations

import math

import pytest
from conftest import make_request

from shadoweval import (
    CompositeScorer,
    ExactMatchScorer,
    LengthPenaltyScorer,
    RubricScorer,
    ScoreDistribution,
    TokenOverlapScorer,
    bootstrap_mean_ci,
    token_f1,
)


def test_token_f1_is_one_for_identical_sequences() -> None:
    assert token_f1(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_token_f1_hand_computed() -> None:
    # predicted = [a, b, b], reference = [a, b]
    # overlap = 2, precision = 2/3, recall = 2/2 = 1
    # F1 = 2 * (2/3) * 1 / (2/3 + 1) = 4/3 / 5/3 = 0.8
    assert token_f1(["a", "b", "b"], ["a", "b"]) == pytest.approx(0.8)


def test_token_f1_hand_computed_with_disjoint_tokens() -> None:
    assert token_f1(["x", "y"], ["a", "b"]) == 0.0


def test_token_f1_counts_multiset_overlap() -> None:
    # predicted = [a, a, b], reference = [a, b, b] -> overlap 2
    # precision = 2/3, recall = 2/3, F1 = 2/3
    assert token_f1(["a", "a", "b"], ["a", "b", "b"]) == pytest.approx(2.0 / 3.0)


def test_token_f1_empty_sequences() -> None:
    assert token_f1([], []) == 1.0
    assert token_f1(["a"], []) == 0.0
    assert token_f1([], ["a"]) == 0.0


def test_exact_match_scorer() -> None:
    request = make_request(0, response="Your invoice is in Settings.")
    scorer = ExactMatchScorer()
    assert scorer.score(request, "Your invoice is in Settings.") == 1.0
    assert scorer.score(request, "your invoice is in settings.") == 1.0
    assert scorer.score(request, "Something else entirely") == 0.0


def test_exact_match_scorer_with_explicit_expected() -> None:
    request = make_request(0, response="production answer")
    scorer = ExactMatchScorer(expected="golden answer")
    assert scorer.score(request, "golden answer") == 1.0
    assert scorer.score(request, "production answer") == 0.0


def test_token_overlap_scorer_matches_manual_f1() -> None:
    request = make_request(0, response="the fee is twelve dollars")
    scorer = TokenOverlapScorer()
    score = scorer.score(request, "the fee is twelve dollars")
    assert score == pytest.approx(1.0)
    partial = scorer.score(request, "the fee is twelve euros")
    assert 0.0 < partial < 1.0


def test_rubric_scorer_counts_keyword_coverage() -> None:
    request = make_request(0, response="irrelevant")
    scorer = RubricScorer(keywords=("refund", "invoice", "policy"))
    assert scorer.score(request, "here is your refund and invoice") == pytest.approx(2.0 / 3.0)
    assert scorer.score(request, "refund invoice policy") == 1.0
    assert scorer.score(request, "nothing relevant") == 0.0


def test_rubric_scorer_matches_substrings() -> None:
    request = make_request(0, response="x")
    scorer = RubricScorer(keywords=("refund",))
    assert scorer.score(request, "we issued refunds yesterday") == 1.0


def test_rubric_scorer_without_keywords_uses_the_reference() -> None:
    request = make_request(0, response="reset password email")
    scorer = RubricScorer()
    assert scorer.score(request, "reset password email") == 1.0
    assert scorer.score(request, "reset") == pytest.approx(1.0 / 3.0)


def test_length_penalty_scorer_penalises_deviation() -> None:
    request = make_request(0, response="one two three four five six")
    scorer = LengthPenaltyScorer(tolerance=0.2)
    assert scorer.score(request, "one two three four five six") == 1.0
    assert scorer.score(request, "one two three four five") == 1.0
    assert scorer.score(request, "one") < 1.0
    assert scorer.score(request, "one two") < 1.0


def test_length_penalty_scorer_is_bounded() -> None:
    request = make_request(0, response="one two three")
    scorer = LengthPenaltyScorer()
    assert 0.0 <= scorer.score(request, "") <= 1.0
    assert 0.0 <= scorer.score(request, "a b c d e f g h i j k l m n o p") <= 1.0


def test_composite_scorer_weighting() -> None:
    request = make_request(0, response="unrelated production answer")
    exact = ExactMatchScorer()
    rubric = RubricScorer(keywords=("refund", "policy"))
    composite = CompositeScorer(scorers=(exact, rubric), weights=(0.25, 0.75))
    # exact = 0, rubric = 1 -> 0 * 0.25 + 1 * 0.75
    assert composite.score(request, "refund policy") == pytest.approx(0.75)


def test_composite_scorer_normalises_weights() -> None:
    request = make_request(0, response="unrelated production answer")
    composite = CompositeScorer(
        scorers=(ExactMatchScorer(), RubricScorer(keywords=("refund", "policy"))),
        weights=(1.0, 3.0),
    )
    assert composite.weights == (0.25, 0.75)
    assert composite.score(request, "refund policy") == pytest.approx(0.75)


def test_composite_scorer_rejects_mismatched_weights() -> None:
    with pytest.raises(ValueError, match="weights length"):
        CompositeScorer(scorers=(ExactMatchScorer(),), weights=(0.5, 0.5))


def test_composite_scorer_rejects_empty_scorers() -> None:
    with pytest.raises(ValueError, match="at least one sub-scorer"):
        CompositeScorer(scorers=())


def test_composite_scorer_rejects_all_zero_weights() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        CompositeScorer(scorers=(ExactMatchScorer(),), weights=(0.0,))


def test_distribution_mean_and_standard_deviation() -> None:
    distribution = ScoreDistribution.from_scores([0.1, 0.2, 0.3, 0.4])
    assert distribution.n == 4
    assert distribution.mean == pytest.approx(0.25)
    assert distribution.std == pytest.approx(math.sqrt(0.05 / 3.0))


def test_distribution_percentiles() -> None:
    distribution = ScoreDistribution.from_scores([0.0, 0.25, 0.5, 0.75, 1.0])
    assert distribution.percentile(0.0) == 0.0
    assert distribution.percentile(50.0) == 0.5
    assert distribution.percentile(100.0) == 1.0
    assert distribution.minimum == 0.0
    assert distribution.maximum == 1.0


def test_distribution_percentile_of_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty distribution"):
        ScoreDistribution.from_scores([]).percentile(50.0)


def test_bootstrap_interval_contains_the_mean() -> None:
    scores = [0.4, 0.5, 0.6, 0.45, 0.55, 0.7, 0.35, 0.65, 0.5, 0.5]
    distribution = ScoreDistribution.from_scores(scores, seed=3)
    interval = distribution.confidence_interval
    assert interval is not None
    assert interval.contains(distribution.mean)


def test_bootstrap_interval_is_deterministic() -> None:
    scores = [0.4, 0.5, 0.6, 0.45, 0.55, 0.7, 0.35, 0.65]
    first = bootstrap_mean_ci(scores, seed=11)
    second = bootstrap_mean_ci(scores, seed=11)
    assert (first.lower, first.upper) == (second.lower, second.upper)


def test_bootstrap_interval_narrows_with_more_samples() -> None:
    rng_values = [0.5 + 0.1 * ((index * 37) % 11) / 11.0 for index in range(200)]
    small = bootstrap_mean_ci(rng_values[:20], seed=5)
    large = bootstrap_mean_ci(rng_values, seed=5)
    assert large.width < small.width


def test_bootstrap_interval_handles_a_single_sample() -> None:
    interval = bootstrap_mean_ci([0.42], seed=1)
    assert interval.lower == interval.upper == pytest.approx(0.42)


def test_bootstrap_rejects_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError, match="confidence"):
        bootstrap_mean_ci([1.0], confidence=1.5)
    with pytest.raises(ValueError, match="resamples"):
        bootstrap_mean_ci([1.0], resamples=0)


def test_empty_distribution_is_safe() -> None:
    distribution = ScoreDistribution.from_scores([])
    assert distribution.n == 0
    assert distribution.mean == 0.0
    assert distribution.confidence_interval is None


def test_exact_match_scorer_without_normalisation() -> None:
    request = make_request(0, response="Exact Answer")
    scorer = ExactMatchScorer(normalize=False)
    assert scorer.score(request, "Exact Answer") == 1.0
    assert scorer.score(request, "exact answer") == 0.0


def test_rubric_scorer_with_no_keywords_and_no_reference_tokens() -> None:
    request = make_request(0, response="!!!")
    assert RubricScorer().score(request, "anything") == 0.0


def test_length_penalty_scorer_handles_two_empty_answers() -> None:
    request = make_request(0, response="")
    scorer = LengthPenaltyScorer()
    assert scorer.score(request, "") == 1.0


def test_composite_scorer_defaults_to_equal_weights() -> None:
    composite = CompositeScorer(
        scorers=(ExactMatchScorer(), RubricScorer(keywords=("refund", "policy")))
    )
    assert composite.weights == (0.5, 0.5)
