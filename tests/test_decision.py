"""Tests for the statistics: the t distribution, the paired t-test and the verdict."""

from __future__ import annotations

import math

import pytest

from shadoweval import (
    Decision,
    PairedTTest,
    RegressionAnalyzer,
    ReplayResult,
    TTestResult,
    WilsonInterval,
    cohens_d,
    regularized_incomplete_beta,
    student_t_cdf,
    student_t_ppf,
    student_t_pvalue,
)


def _result(
    index: int,
    baseline: float,
    candidate: float,
    *,
    baseline_cost: float = 0.001,
    candidate_cost: float = 0.001,
    baseline_latency: float = 900.0,
    candidate_latency: float = 900.0,
) -> ReplayResult:
    """Build a :class:`ReplayResult` for the decision tests."""
    return ReplayResult(
        request_id=f"req-{index:04d}",
        baseline_response="baseline",
        candidate_response="candidate",
        baseline_score=baseline,
        candidate_score=candidate,
        latency_delta_ms=candidate_latency - baseline_latency,
        cost_delta_usd=candidate_cost - baseline_cost,
        diverged=False,
        baseline_cost_usd=baseline_cost,
        candidate_cost_usd=candidate_cost,
        baseline_latency_ms=baseline_latency,
        candidate_latency_ms=candidate_latency,
    )


# --------------------------------------------------------------------------
# Student's t distribution
# --------------------------------------------------------------------------


def test_student_t_pvalue_against_known_critical_value() -> None:
    # t = 2.0 with 10 degrees of freedom -> two-sided p = 0.073388.
    assert student_t_pvalue(2.0, 10) == pytest.approx(0.0734, abs=5e-4)


def test_student_t_pvalue_for_t_2_with_nine_degrees_of_freedom() -> None:
    # t = 2.0 with 9 degrees of freedom -> two-sided p = 0.076553.
    assert student_t_pvalue(2.0, 9) == pytest.approx(0.0766, abs=5e-4)


def test_student_t_pvalue_at_the_five_percent_critical_value() -> None:
    # t = 2.262 with 9 degrees of freedom is the 5% two-sided critical value.
    assert student_t_pvalue(2.262, 9) == pytest.approx(0.05, abs=1e-3)


def test_student_t_pvalue_at_the_one_percent_critical_value() -> None:
    # t = 3.250 with 9 degrees of freedom is the 1% two-sided critical value.
    assert student_t_pvalue(3.250, 9) == pytest.approx(0.01, abs=1e-3)


def test_student_t_pvalue_at_the_ten_percent_critical_value() -> None:
    # t = 1.833 with 9 degrees of freedom is the 10% two-sided critical value.
    assert student_t_pvalue(1.833, 9) == pytest.approx(0.10, abs=1e-3)


def test_student_t_cdf_is_symmetric_and_centred() -> None:
    assert student_t_cdf(0.0, 9) == pytest.approx(0.5)
    assert student_t_cdf(2.0, 9) + student_t_cdf(-2.0, 9) == pytest.approx(1.0)


def test_student_t_cdf_matches_the_regularised_beta() -> None:
    x = 9.0 / (9.0 + 4.0)
    assert student_t_cdf(2.0, 9) == pytest.approx(
        1.0 - 0.5 * regularized_incomplete_beta(4.5, 0.5, x)
    )


def test_student_t_ppf_inverts_the_cdf() -> None:
    assert student_t_ppf(0.975, 9) == pytest.approx(2.262157, abs=1e-5)
    assert student_t_ppf(0.95, 9) == pytest.approx(1.833113, abs=1e-5)
    assert student_t_cdf(student_t_ppf(0.975, 9), 9) == pytest.approx(0.975)


def test_student_t_ppf_of_the_median_is_zero() -> None:
    assert student_t_ppf(0.5, 5) == 0.0


def test_student_t_functions_reject_invalid_degrees_of_freedom() -> None:
    with pytest.raises(ValueError, match="df must be positive"):
        student_t_cdf(1.0, 0.0)
    with pytest.raises(ValueError, match="df must be positive"):
        student_t_pvalue(1.0, -1.0)
    with pytest.raises(ValueError, match="p must be in"):
        student_t_ppf(0.0, 5)


def test_regularized_incomplete_beta_known_values() -> None:
    # I_x(a, b) for a = b = 1 is just x; for a = b = 2 it is 3x^2 - 2x^3.
    assert regularized_incomplete_beta(1.0, 1.0, 0.37) == pytest.approx(0.37)
    assert regularized_incomplete_beta(2.0, 2.0, 0.4) == pytest.approx(3 * 0.16 - 2 * 0.064)
    assert regularized_incomplete_beta(0.5, 0.5, 0.5) == pytest.approx(0.5)
    assert regularized_incomplete_beta(2.0, 3.0, 0.0) == 0.0
    assert regularized_incomplete_beta(2.0, 3.0, 1.0) == 1.0


def test_regularized_incomplete_beta_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        regularized_incomplete_beta(0.0, 1.0, 0.5)


# --------------------------------------------------------------------------
# Paired t-test
# --------------------------------------------------------------------------


def test_paired_t_test_of_identical_data_gives_p_one() -> None:
    baseline = [0.5, 0.6, 0.4, 0.7, 0.55, 0.65, 0.45, 0.6]
    result = PairedTTest().run(baseline, list(baseline))
    assert result.p_value == pytest.approx(1.0, abs=1e-9)
    assert result.t_statistic == pytest.approx(0.0)
    assert result.mean_difference == pytest.approx(0.0)
    assert result.df == 7


def test_paired_t_test_hand_computed_t_statistic() -> None:
    baseline = [1.0, 2.0, 3.0, 4.0, 5.0]
    candidate = [2.0, 3.0, 4.0, 5.0, 6.0]
    # differences are all +1 -> mean 1, std 0 -> t is infinite
    result = PairedTTest().run(baseline, candidate)
    assert result.mean_difference == pytest.approx(1.0)
    assert math.isinf(result.t_statistic)
    assert result.p_value == 0.0


def test_paired_t_test_hand_computed_statistics() -> None:
    baseline = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    candidate = [0.2, 0.2, 0.4, 0.4, 0.6, 0.6, 0.8, 0.8, 1.0, 1.0]
    differences = [c - b for b, c in zip(baseline, candidate, strict=True)]
    expected_mean = sum(differences) / len(differences)
    expected_std = math.sqrt(
        sum((d - expected_mean) ** 2 for d in differences) / (len(differences) - 1)
    )
    expected_t = expected_mean / (expected_std / math.sqrt(len(differences)))
    result = PairedTTest().run(baseline, candidate)
    assert result.mean_difference == pytest.approx(expected_mean)
    assert result.std_difference == pytest.approx(expected_std)
    assert result.t_statistic == pytest.approx(expected_t)
    assert result.p_value == pytest.approx(student_t_pvalue(expected_t, 9))
    assert result.n == 10
    assert result.df == 9


def test_paired_t_test_detects_a_clearly_better_candidate() -> None:
    baseline = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50, 0.52, 0.48]
    candidate = [value + 0.15 for value in baseline]
    result = PairedTTest().run(baseline, candidate)
    assert result.p_value < 0.01
    assert result.mean_difference == pytest.approx(0.15)
    assert result.significant is True


def test_paired_t_test_detects_a_clearly_worse_candidate() -> None:
    baseline = [0.50, 0.52, 0.48, 0.51, 0.49, 0.53, 0.47, 0.50, 0.52, 0.48]
    candidate = [value - 0.15 for value in baseline]
    result = PairedTTest().run(baseline, candidate)
    assert result.p_value < 0.01
    assert result.mean_difference == pytest.approx(-0.15)
    assert result.significant is True


def test_paired_t_test_confidence_interval_brackets_the_mean() -> None:
    baseline = [0.5] * 10
    candidate = [0.5 + 0.01 * (index - 5) for index in range(10)]
    result = PairedTTest().run(baseline, candidate)
    assert result.confidence_interval is not None
    assert result.confidence_interval.contains(result.mean_difference)


def test_paired_t_test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        PairedTTest().run([0.1, 0.2], [0.1])


def test_paired_t_test_rejects_too_few_observations() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        PairedTTest().run([0.1], [0.2])


def test_paired_t_test_rejects_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="alpha"):
        PairedTTest(alpha=1.5)


# --------------------------------------------------------------------------
# Cohen's d
# --------------------------------------------------------------------------


def test_cohens_d_hand_computed() -> None:
    differences = [0.2, 0.4, 0.6, 0.8, 1.0]
    mean = 0.6
    std = math.sqrt(sum((d - mean) ** 2 for d in differences) / 4)
    assert cohens_d(differences) == pytest.approx(mean / std)


def test_cohens_d_is_zero_for_zero_differences() -> None:
    assert cohens_d([0.0, 0.0, 0.0, 0.0]) == 0.0


def test_cohens_d_is_infinite_for_a_constant_shift() -> None:
    assert math.isinf(cohens_d([1.0, 1.0, 1.0]))


def test_cohens_d_rejects_an_empty_sample() -> None:
    with pytest.raises(ValueError, match="empty sample"):
        cohens_d([])


# --------------------------------------------------------------------------
# Wilson interval
# --------------------------------------------------------------------------


def test_wilson_interval_known_value() -> None:
    # 0 successes out of 20 at 95% confidence.
    interval = WilsonInterval.compute(0, 20)
    assert interval.lower == pytest.approx(0.0, abs=1e-9)
    assert interval.upper == pytest.approx(0.1614, abs=2e-3)


def test_wilson_interval_hand_computed() -> None:
    interval = WilsonInterval.compute(10, 20)
    z = 1.959963984540054
    proportion = 0.5
    denominator = 1.0 + z * z / 20
    center = (proportion + z * z / 40) / denominator
    margin = z / denominator * math.sqrt(proportion * (1 - proportion) / 20 + z * z / (4 * 400))
    assert interval.lower == pytest.approx(center - margin)
    assert interval.upper == pytest.approx(center + margin)
    assert interval.proportion == 0.5
    assert interval.contains(0.5)


def test_wilson_interval_stays_inside_the_unit_interval() -> None:
    for successes, total in ((0, 5), (5, 5), (3, 7)):
        interval = WilsonInterval.compute(successes, total)
        assert 0.0 <= interval.lower <= interval.upper <= 1.0


def test_wilson_interval_of_an_empty_sample() -> None:
    interval = WilsonInterval.compute(0, 0)
    assert interval.lower == 0.0
    assert interval.upper == 1.0


def test_wilson_interval_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="successes must be in"):
        WilsonInterval.compute(5, 3)
    with pytest.raises(ValueError, match="total must be non-negative"):
        WilsonInterval.compute(0, -1)
    with pytest.raises(ValueError, match="confidence"):
        WilsonInterval.compute(1, 2, confidence=0.0)


# --------------------------------------------------------------------------
# Regression analyzer
# --------------------------------------------------------------------------


def test_analyzer_blocks_a_significantly_worse_candidate() -> None:
    results = [_result(index, 0.80, 0.60) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10).analyze(results)
    assert verdict.decision is Decision.BLOCK
    assert verdict.p_value < 0.05
    assert verdict.mean_delta < 0.0
    assert verdict.sample_size == 30
    assert any("significantly worse" in reason for reason in verdict.reasons)


def test_analyzer_blocks_when_the_quality_drop_is_large() -> None:
    results = [_result(index, 0.80, 0.70) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10, min_effect_size=0.05).analyze(results)
    assert verdict.decision is Decision.BLOCK
    assert any("minimum effect size" in reason for reason in verdict.reasons)


def test_analyzer_ships_an_equal_candidate() -> None:
    results = [_result(index, 0.70, 0.70) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10).analyze(results)
    assert verdict.decision is Decision.SHIP
    assert verdict.p_value == pytest.approx(1.0, abs=1e-9)
    assert verdict.mean_delta == pytest.approx(0.0)
    assert verdict.blocked is False


def test_analyzer_ships_a_significantly_better_candidate() -> None:
    results = [_result(index, 0.60, 0.75) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10).analyze(results)
    assert verdict.decision is Decision.SHIP
    assert verdict.p_value < 0.05
    assert verdict.mean_delta > 0.0


def test_analyzer_is_inconclusive_on_a_small_sample() -> None:
    results = [_result(index, 0.80, 0.40) for index in range(5)]
    verdict = RegressionAnalyzer(min_samples=20).analyze(results)
    assert verdict.decision is Decision.INCONCLUSIVE
    assert verdict.sample_size == 5
    assert any("at least 20" in reason for reason in verdict.reasons)


def test_analyzer_is_inconclusive_without_results() -> None:
    verdict = RegressionAnalyzer().analyze([])
    assert verdict.decision is Decision.INCONCLUSIVE
    assert verdict.sample_size == 0
    assert verdict.reasons


def test_analyzer_blocks_on_the_cost_criterion() -> None:
    results = [
        _result(index, 0.70, 0.70, baseline_cost=0.001, candidate_cost=0.003) for index in range(30)
    ]
    verdict = RegressionAnalyzer(min_samples=10, max_cost_ratio=1.25).analyze(results)
    assert verdict.decision is Decision.BLOCK
    assert verdict.cost_delta == pytest.approx(0.002)
    assert any("cost" in reason for reason in verdict.reasons)


def test_analyzer_blocks_on_the_absolute_cost_criterion() -> None:
    results = [
        _result(index, 0.70, 0.70, baseline_cost=0.001, candidate_cost=0.0015)
        for index in range(30)
    ]
    verdict = RegressionAnalyzer(
        min_samples=10, max_cost_ratio=10.0, max_cost_delta_usd=0.0001
    ).analyze(results)
    assert verdict.decision is Decision.BLOCK
    assert any("more per request" in reason for reason in verdict.reasons)


def test_analyzer_blocks_on_the_latency_criterion() -> None:
    results = [
        _result(index, 0.70, 0.70, baseline_latency=500.0, candidate_latency=1500.0)
        for index in range(30)
    ]
    verdict = RegressionAnalyzer(min_samples=10, max_latency_ratio=1.25).analyze(results)
    assert verdict.decision is Decision.BLOCK
    assert verdict.latency_delta == pytest.approx(1000.0)
    assert any("latency" in reason for reason in verdict.reasons)


def test_analyzer_allows_small_cost_and_latency_drift() -> None:
    results = [
        _result(index, 0.70, 0.70, baseline_cost=0.001, candidate_cost=0.0011)
        for index in range(30)
    ]
    verdict = RegressionAnalyzer(min_samples=10, max_cost_ratio=1.25).analyze(results)
    assert verdict.decision is Decision.SHIP


def test_verdict_summary_is_human_readable() -> None:
    results = [_result(index, 0.80, 0.60) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10).analyze(results)
    summary = verdict.summary()
    assert summary.startswith("BLOCK:")
    assert "p=" in summary
    assert "d=" in summary


def test_analyzer_reports_bootstrap_and_t_intervals() -> None:
    results = [_result(index, 0.80, 0.65 + 0.01 * (index % 5)) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10).analyze(results)
    assert verdict.confidence_interval is not None
    assert verdict.bootstrap_interval is not None
    assert verdict.bootstrap_interval.contains(verdict.mean_delta)
    assert verdict.effect_size < 0.0


def test_t_test_result_exposes_the_effect_size() -> None:
    result: TTestResult = PairedTTest().run([0.5, 0.6, 0.7], [0.6, 0.7, 0.8])
    assert result.effect_size == pytest.approx(cohens_d(result.differences))


def test_wilson_interval_centre_and_width() -> None:
    interval = WilsonInterval.compute(10, 20)
    assert interval.center == pytest.approx(0.5 * (interval.lower + interval.upper))
    assert interval.width == pytest.approx(interval.upper - interval.lower)


def test_verdict_helpers_report_the_decision() -> None:
    results = [_result(index, 0.80, 0.60) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10).analyze(results)
    assert verdict.blocked is True
    assert verdict.shipped is False
    safe = RegressionAnalyzer(min_samples=10).analyze(
        [_result(index, 0.70, 0.70) for index in range(30)]
    )
    assert safe.shipped is True
    assert safe.blocked is False


def test_paired_t_test_p_value_of_a_single_observation_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least two observations"):
        PairedTTest().p_value([0.1])


def test_paired_t_test_rejects_an_invalid_confidence() -> None:
    with pytest.raises(ValueError, match="confidence"):
        PairedTTest(confidence=0.0)


def test_student_t_pvalue_of_an_infinite_statistic_is_zero() -> None:
    assert student_t_pvalue(math.inf, 9) == 0.0
    assert student_t_pvalue(-math.inf, 9) == 0.0


def test_normal_quantile_covers_the_tails() -> None:
    from shadoweval.decision import _normal_quantile

    assert _normal_quantile(0.975) == pytest.approx(1.959963984540054, abs=1e-8)
    assert _normal_quantile(0.025) == pytest.approx(-1.959963984540054, abs=1e-8)
    assert _normal_quantile(0.5) == pytest.approx(0.0, abs=1e-9)
    assert _normal_quantile(0.001) < -3.0


def test_normal_quantile_rejects_invalid_probabilities() -> None:
    from shadoweval.decision import _normal_quantile

    with pytest.raises(ValueError, match="p must be in"):
        _normal_quantile(0.0)


def test_analyzer_reports_a_clean_run() -> None:
    results = [_result(index, 0.70, 0.70) for index in range(30)]
    verdict = RegressionAnalyzer(min_samples=10).analyze(results)
    assert verdict.reasons
    assert "no statistically significant" in verdict.reasons[0]
