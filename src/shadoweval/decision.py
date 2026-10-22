"""Statistical machinery for deciding whether a candidate may ship.

The decision rests on a *paired* t-test over per-request score differences:
every production request contributes one observation to both the baseline and
the candidate, so the observations are naturally paired and the paired test is
far more sensitive than comparing two independent means.

The Student's t distribution is implemented from scratch: the survival function
is expressed through the regularised incomplete beta function, which is
evaluated with a numerically stable continued fraction (Lentz's algorithm)
using :func:`math.lgamma` for the normalisation. No SciPy, no NumPy
special-functions - just the standard library.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from shadoweval.scoring import ConfidenceInterval, bootstrap_mean_ci

_EPS = 3.0e-16
_FPMIN = 1.0e-300
_MAXIT = 300


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Evaluate the continued fraction for the incomplete beta function.

    This is Lentz's algorithm as used in Numerical Recipes: the fraction is
    evaluated with modified Lentz iteration, which converges for every
    ``x`` in ``[0, 1]``.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < _FPMIN:
        d = _FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, _MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = 1.0 + aa / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """The regularised incomplete beta function ``I_x(a, b)``.

    ``I_x(a, b)`` is the probability that a ``Beta(a, b)`` random variable is
    smaller than ``x``. The implementation switches to the symmetric
    formulation near ``x = 1`` to avoid cancellation.
    """
    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    if x < (a + 1.0) / (a + b + 2.0):
        front = math.exp(log_beta + a * math.log(x) + b * math.log1p(-x))
        return front * _beta_continued_fraction(a, b, x) / a
    front = math.exp(log_beta + b * math.log1p(-x) + a * math.log(x))
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def student_t_cdf(t: float, df: float) -> float:
    """Cumulative distribution function of the Student's t distribution.

    Args:
        t: The value at which the CDF is evaluated.
        df: Degrees of freedom (must be positive).
    """
    if df <= 0.0:
        raise ValueError(f"df must be positive, got {df}")
    if t == 0.0:
        return 0.5
    x = df / (df + t * t)
    tail = regularized_incomplete_beta(0.5 * df, 0.5, x)
    if t > 0.0:
        return 1.0 - 0.5 * tail
    return 0.5 * tail


def student_t_pvalue(t: float, df: float) -> float:
    """Two-sided p-value of the Student's t statistic ``t`` with ``df`` degrees of freedom."""
    if df <= 0.0:
        raise ValueError(f"df must be positive, got {df}")
    if math.isinf(t):
        return 0.0
    x = df / (df + t * t)
    return regularized_incomplete_beta(0.5 * df, 0.5, x)


def student_t_ppf(p: float, df: float) -> float:
    """Quantile of the Student's t distribution.

    The inverse CDF is found by bisection on :func:`student_t_cdf`, which is
    monotone and numerically stable.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    if df <= 0.0:
        raise ValueError(f"df must be positive, got {df}")
    if p == 0.5:
        return 0.0
    low, high = -1.0e3, 1.0e3
    for _ in range(200):
        mid = 0.5 * (low + high)
        if student_t_cdf(mid, df) < p:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def cohens_d(differences: Sequence[float]) -> float:
    """Cohen's d for paired samples.

    For paired data the effect size is the mean difference divided by the
    standard deviation of the differences. A zero standard deviation yields
    ``0.0`` for a zero mean and an infinite (signed) effect otherwise.
    """
    values = np.asarray(list(differences), dtype=np.float64)
    if values.size == 0:
        raise ValueError("cannot compute an effect size of an empty sample")
    mean = float(values.mean())
    if values.size < 2:
        return 0.0
    std = float(values.std(ddof=1))
    if std == 0.0:
        if mean == 0.0:
            return 0.0
        return math.copysign(math.inf, mean)
    return mean / std


@dataclass(slots=True)
class TTestResult:
    """Outcome of a paired t-test."""

    t_statistic: float
    p_value: float
    df: int
    mean_difference: float
    std_difference: float
    n: int
    confidence_interval: ConfidenceInterval | None = None
    significant: bool = False

    @property
    def effect_size(self) -> float:
        """Cohen's d of the paired differences."""
        return cohens_d(self.differences) if self.n else 0.0

    differences: tuple[float, ...] = field(default=())


@dataclass(slots=True)
class PairedTTest:
    """A paired Student's t-test implemented from first principles.

    For ``n`` paired observations the differences ``d_i = candidate_i -
    baseline_i`` are summarised by

    .. code-block:: text

        t = mean(d) / (std(d) / sqrt(n))

    and the two-sided p-value comes from the Student's t distribution with
    ``n - 1`` degrees of freedom.
    """

    alpha: float = 0.05
    confidence: float = 0.95

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {self.confidence}")

    def differences(self, baseline: Sequence[float], candidate: Sequence[float]) -> np.ndarray:
        """Per-item differences ``candidate - baseline``."""
        if len(baseline) != len(candidate):
            raise ValueError(
                f"baseline and candidate must have the same length, "
                f"got {len(baseline)} and {len(candidate)}"
            )
        return np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)

    def t_statistic(self, differences: Sequence[float]) -> float:
        """The t statistic of the paired differences."""
        values = np.asarray(list(differences), dtype=np.float64)
        count = int(values.size)
        if count < 2:
            raise ValueError("a paired t-test needs at least two observations")
        std = float(values.std(ddof=1))
        if std == 0.0:
            mean = float(values.mean())
            if mean == 0.0:
                return 0.0
            return math.copysign(math.inf, mean)
        return float(values.mean()) / (std / math.sqrt(count))

    def p_value(self, differences: Sequence[float]) -> float:
        """The two-sided p-value of the paired differences."""
        values = np.asarray(list(differences), dtype=np.float64)
        count = int(values.size)
        if count < 2:
            raise ValueError("a paired t-test needs at least two observations")
        t = self.t_statistic(values)
        if math.isinf(t):
            return 0.0
        return student_t_pvalue(t, count - 1)

    def run(self, baseline: Sequence[float], candidate: Sequence[float]) -> TTestResult:
        """Run the paired t-test over ``baseline`` and ``candidate`` scores."""
        diffs = self.differences(baseline, candidate)
        count = int(diffs.size)
        if count < 2:
            raise ValueError("a paired t-test needs at least two observations")
        mean = float(diffs.mean())
        std = float(diffs.std(ddof=1))
        t_stat = self.t_statistic(diffs)
        p_value = 0.0 if math.isinf(t_stat) else student_t_pvalue(t_stat, count - 1)
        df = count - 1
        if std == 0.0:
            interval = ConfidenceInterval(lower=mean, upper=mean, confidence=self.confidence)
        else:
            margin = student_t_ppf(1.0 - (1.0 - self.confidence) / 2.0, df) * std / math.sqrt(count)
            interval = ConfidenceInterval(
                lower=mean - margin, upper=mean + margin, confidence=self.confidence
            )
        return TTestResult(
            t_statistic=t_stat,
            p_value=p_value,
            df=df,
            mean_difference=mean,
            std_difference=std,
            n=count,
            confidence_interval=interval,
            significant=p_value < self.alpha,
            differences=tuple(float(value) for value in diffs),
        )


@dataclass(slots=True)
class WilsonInterval:
    """Wilson score interval for a single proportion.

    The Wilson interval is preferred over the normal approximation because it
    stays inside ``[0, 1]`` and behaves well for small samples and for
    proportions close to 0 or 1.
    """

    successes: int
    total: int
    confidence: float = 0.95
    lower: float = 0.0
    upper: float = 0.0

    @classmethod
    def compute(cls, successes: int, total: int, *, confidence: float = 0.95) -> WilsonInterval:
        """Compute the Wilson interval for ``successes`` out of ``total``."""
        if total < 0:
            raise ValueError(f"total must be non-negative, got {total}")
        if successes < 0 or successes > total:
            raise ValueError(f"successes must be in [0, {total}], got {successes}")
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence must be in (0, 1), got {confidence}")
        if total == 0:
            return cls(successes=0, total=0, confidence=confidence, lower=0.0, upper=1.0)
        z = _normal_quantile(1.0 - (1.0 - confidence) / 2.0)
        proportion = successes / total
        denominator = 1.0 + z * z / total
        center = (proportion + z * z / (2.0 * total)) / denominator
        margin = (
            z
            / denominator
            * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        )
        return cls(
            successes=successes,
            total=total,
            confidence=confidence,
            lower=max(0.0, center - margin),
            upper=min(1.0, center + margin),
        )

    @property
    def proportion(self) -> float:
        """The observed proportion."""
        return self.successes / self.total if self.total else 0.0

    @property
    def center(self) -> float:
        """Midpoint of the interval."""
        return 0.5 * (self.lower + self.upper)

    @property
    def width(self) -> float:
        """Width of the interval."""
        return self.upper - self.lower

    def contains(self, proportion: float) -> bool:
        """Whether ``proportion`` lies inside the interval."""
        return self.lower <= proportion <= self.upper


def _normal_quantile(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p > 1.0 - p_low:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


class Decision(str, Enum):
    """The verdict of a shadow evaluation."""

    SHIP = "ship"
    """The candidate is safe to ship."""

    BLOCK = "block"
    """The candidate regressed and must not ship."""

    INCONCLUSIVE = "inconclusive"
    """Not enough evidence either way."""


@dataclass(slots=True)
class Verdict:
    """The full explanation of a shadow evaluation decision."""

    decision: Decision
    p_value: float
    mean_delta: float
    effect_size: float
    confidence_interval: ConfidenceInterval | None
    cost_delta: float
    latency_delta: float
    reasons: tuple[str, ...] = ()
    sample_size: int = 0
    bootstrap_interval: ConfidenceInterval | None = None
    t_statistic: float = 0.0

    @property
    def shipped(self) -> bool:
        """Whether the candidate may ship."""
        return self.decision is Decision.SHIP

    @property
    def blocked(self) -> bool:
        """Whether the candidate must not ship."""
        return self.decision is Decision.BLOCK

    def summary(self) -> str:
        """A one-line human-readable summary."""
        first = self.reasons[0] if self.reasons else "no additional details"
        return (
            f"{self.decision.value.upper()}: p={self.p_value:.4f}, "
            f"delta={self.mean_delta:+.4f}, d={self.effect_size:+.3f} "
            f"(n={self.sample_size}) - {first}"
        )


@dataclass(slots=True)
class RegressionAnalyzer:
    """Turns replay results into a ship / block / inconclusive verdict.

    Args:
        paired_ttest: The paired t-test used to test for a quality change.
        min_effect_size: The smallest score drop that matters in practice.
            A drop smaller than this is treated as noise even if the p-value
            is small.
        alpha: Significance level of the t-test.
        min_samples: Minimum number of replayed requests for a verdict.
        max_cost_ratio: Largest tolerated candidate/baseline cost ratio.
        max_latency_ratio: Largest tolerated candidate/baseline latency ratio.
        max_cost_delta_usd: Largest tolerated absolute per-request cost increase.
        max_latency_delta_ms: Largest tolerated absolute latency increase.
    """

    paired_ttest: PairedTTest = field(default_factory=PairedTTest)
    min_effect_size: float = 0.01
    alpha: float = 0.05
    min_samples: int = 20
    max_cost_ratio: float = 1.25
    max_latency_ratio: float = 1.25
    max_cost_delta_usd: float | None = None
    max_latency_delta_ms: float | None = None
    bootstrap_resamples: int = 1000
    bootstrap_seed: int = 0

    def analyze(self, replay_results: Sequence[object]) -> Verdict:
        """Analyse ``replay_results`` and return a :class:`Verdict`.

        ``replay_results`` are :class:`~shadoweval.replay.ReplayResult` objects;
        any object exposing ``baseline_score``, ``candidate_score``,
        ``cost_delta_usd`` and ``latency_delta_ms`` works.
        """
        results = list(replay_results)
        sample_size = len(results)
        if sample_size == 0:
            return Verdict(
                decision=Decision.INCONCLUSIVE,
                p_value=1.0,
                mean_delta=0.0,
                effect_size=0.0,
                confidence_interval=None,
                cost_delta=0.0,
                latency_delta=0.0,
                reasons=("no replay results to analyse",),
                sample_size=0,
            )

        baseline = [float(result.baseline_score) for result in results]  # type: ignore[attr-defined]
        candidate = [float(result.candidate_score) for result in results]  # type: ignore[attr-defined]
        cost_deltas = [float(result.cost_delta_usd) for result in results]  # type: ignore[attr-defined]
        latency_deltas = [float(result.latency_delta_ms) for result in results]  # type: ignore[attr-defined]

        mean_cost_delta = float(np.mean(cost_deltas))
        mean_latency_delta = float(np.mean(latency_deltas))

        if sample_size < self.min_samples:
            return Verdict(
                decision=Decision.INCONCLUSIVE,
                p_value=1.0,
                mean_delta=0.0,
                effect_size=0.0,
                confidence_interval=None,
                cost_delta=mean_cost_delta,
                latency_delta=mean_latency_delta,
                reasons=(
                    f"only {sample_size} replayed requests, "
                    f"at least {self.min_samples} are required for a verdict",
                ),
                sample_size=sample_size,
            )

        differences = [
            candidate_value - baseline_value
            for baseline_value, candidate_value in zip(baseline, candidate, strict=True)
        ]
        test = self.paired_ttest.run(baseline, candidate)
        mean_delta = test.mean_difference
        effect_size = cohens_d(differences)
        bootstrap = bootstrap_mean_ci(
            np.asarray(differences, dtype=np.float64),
            confidence=self.paired_ttest.confidence,
            resamples=self.bootstrap_resamples,
            seed=self.bootstrap_seed,
        )

        reasons: list[str] = []
        decision: Decision

        cost_problem = self._cost_problem(results, mean_cost_delta)
        latency_problem = self._latency_problem(results, mean_latency_delta)

        significant = test.p_value < self.alpha
        if significant and mean_delta < 0.0:
            decision = Decision.BLOCK
            reasons.append(
                f"candidate scored significantly worse than baseline "
                f"(p={test.p_value:.4g} < {self.alpha}, mean delta {mean_delta:+.4f})"
            )
            if abs(mean_delta) >= self.min_effect_size:
                reasons.append(
                    f"quality drop {abs(mean_delta):.4f} exceeds the minimum effect size "
                    f"{self.min_effect_size:.4f}"
                )
        elif significant and mean_delta > 0.0:
            decision = Decision.SHIP
            reasons.append(
                f"candidate scored significantly better than baseline "
                f"(p={test.p_value:.4g} < {self.alpha}, mean delta {mean_delta:+.4f})"
            )
        else:
            decision = Decision.SHIP
            reasons.append(
                f"no statistically significant quality change detected "
                f"(p={test.p_value:.4g} >= {self.alpha}, mean delta {mean_delta:+.4f})"
            )

        if cost_problem:
            decision = Decision.BLOCK
            reasons.append(cost_problem)
        if latency_problem:
            decision = Decision.BLOCK
            reasons.append(latency_problem)
        if decision is Decision.SHIP and not reasons:
            reasons.append("no regressions detected")

        return Verdict(
            decision=decision,
            p_value=test.p_value,
            mean_delta=mean_delta,
            effect_size=effect_size,
            confidence_interval=test.confidence_interval,
            cost_delta=mean_cost_delta,
            latency_delta=mean_latency_delta,
            reasons=tuple(reasons),
            sample_size=sample_size,
            bootstrap_interval=bootstrap,
            t_statistic=test.t_statistic,
        )

    def _cost_problem(self, results: Sequence[object], mean_cost_delta: float) -> str | None:
        """Describe a cost violation, if any."""
        if self.max_cost_delta_usd is not None and mean_cost_delta > self.max_cost_delta_usd:
            return (
                f"candidate costs ${mean_cost_delta:+.6f} more per request, above the "
                f"budget of ${self.max_cost_delta_usd:.6f}"
            )
        baseline_costs = [
            float(getattr(result, "baseline_cost_usd", 0.0) or 0.0) for result in results
        ]
        candidate_costs = [
            float(getattr(result, "candidate_cost_usd", 0.0) or 0.0) for result in results
        ]
        mean_baseline = float(np.mean(baseline_costs))
        mean_candidate = float(np.mean(candidate_costs))
        if mean_baseline > 0.0:
            ratio = mean_candidate / mean_baseline
            if ratio > self.max_cost_ratio:
                return (
                    f"candidate cost is {ratio:.2f}x the baseline cost, above the "
                    f"allowed {self.max_cost_ratio:.2f}x"
                )
        return None

    def _latency_problem(self, results: Sequence[object], mean_latency_delta: float) -> str | None:
        """Describe a latency violation, if any."""
        if self.max_latency_delta_ms is not None and mean_latency_delta > self.max_latency_delta_ms:
            return (
                f"candidate is {mean_latency_delta:+.1f} ms slower per request, above the "
                f"budget of {self.max_latency_delta_ms:.1f} ms"
            )
        baseline_latencies = [
            float(getattr(result, "baseline_latency_ms", 0.0) or 0.0) for result in results
        ]
        candidate_latencies = [
            float(getattr(result, "candidate_latency_ms", 0.0) or 0.0) for result in results
        ]
        mean_baseline = float(np.mean(baseline_latencies))
        mean_candidate = float(np.mean(candidate_latencies))
        if mean_baseline > 0.0:
            ratio = mean_candidate / mean_baseline
            if ratio > self.max_latency_ratio:
                return (
                    f"candidate latency is {ratio:.2f}x the baseline latency, above the "
                    f"allowed {self.max_latency_ratio:.2f}x"
                )
        return None
