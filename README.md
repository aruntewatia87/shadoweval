# shadoweval

> Replay real production traffic against a candidate prompt or model, and decide whether to ship — with a paired t-test, not a hand-curated golden set.

[![CI](https://github.com/aruntewatia87/shadoweval/actions/workflows/ci.yml/badge.svg)](https://github.com/aruntewatia87/shadoweval/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Every prompt-regression tool that ships today requires a golden dataset: a
hand-written file of prompts and the "right" answers, maintained by hand,
checked into the repo, and quietly rotting the moment the product changes.
The teams that skip it ship blind. The teams that build it spend their time
curating data instead of shipping.

Production traffic is the real test set — it is diverse, it is fresh, and it
already contains the failure modes nobody thought to write down. The problem
is that replaying it costs money, so nobody built the tooling. A GitHub search
for `llm shadow traffic production eval` returns **0 repositories**; so does
`llm traffic splitting shadow production`. The seven repos that turn up for
`llm canary prompt deployment` are all 0-star weekend projects. And even the
category leader has not shipped behavioural drift monitoring: Langfuse's
[RFC #12873](https://github.com/langfuse/langfuse/issues/12873) — *"Session-boundary
behavioral drift monitoring"* — is still an open issue.

`shadoweval` is that missing piece: a small, dependency-free library that
takes your captured production traffic, replays a sample of it against a
candidate, and returns a **SHIP / BLOCK / INCONCLUSIVE** verdict backed by
real statistics — a paired t-test, Cohen's *d*, and a bootstrap confidence
interval. The Student's t distribution is implemented from scratch (no SciPy).

---

## ✨ Highlights
- **No golden dataset.** The evaluation set is sampled from the traffic you
  already serve, deduplicated so the sample stays diverse.
- **Four sampling strategies** — `uniform`, `recent`, `stratified` (balanced
  across tenants or tags) and `error_focused` (regressions hide in the
  failures, so those requests are replayed first).
- **Paired statistics, from scratch.** The Student's t CDF is computed with a
  regularised incomplete beta function and Lentz's continued fraction, using
  `math.lgamma` for normalisation. Verified against published critical values
  (t = 2.262, df = 9 → p = 0.05; t = 3.250, df = 9 → p = 0.01).
- **Effect size, not just a p-value.** Cohen's *d* tells you *how much* the
  candidate changed, and a `min_effect_size` floor stops you from blocking a
  release over statistical noise.
- **Cost and latency are gates too.** A candidate that scores the same but
  costs 3x as much is blocked on the cost criterion, with the reason spelled
  out.
- **Hard budget.** `ReplayBudget` stops the replay the moment the dollar or
  request limit is hit and returns a partial result with `truncated=True`.
- **Deterministic end to end.** Seeded sampling, seeded bootstrap, no network,
  no API keys. The same inputs always produce the same verdict.

---

## 🚀 Quickstart
```bash
uv sync --all-extras --dev
uv run pytest -q
```

---

## 🔧 API Tour

### Capture production traffic

```python
from datetime import datetime

from shadoweval import ProductionRequest, TrafficStore

store = TrafficStore()
store.record(
    ProductionRequest(
        request_id="req-00001",
        timestamp=datetime(2026, 3, 1, 9, 0),
        prompt="Where can I download my invoice?",
        model="support-v1",
        response="Your invoice is under Settings and Billing.",
        latency_ms=812.0,
        input_tokens=42,
        output_tokens=88,
        cost_usd=0.0013,
        tenant_id="acme",
        tags=("billing",),
        user_feedback=5.0,
    )
)

# Slice it.
invoices = store.filter_by(tenant_id="acme", tags=("billing",))
failures = store.filter_by(has_error=True)
```

### Sample a diverse evaluation set

```python
from shadoweval import SamplingStrategy

sample = store.sample(
    100,
    seed=20260301,
    strategy=SamplingStrategy.STRATIFIED,
    strata_by="tenant_id",
    dedupe_threshold=0.7,  # drop near-duplicate prompts
)
```

`SamplingStrategy.ERROR_FOCUSED` replays the requests that errored or were
rated badly first — that is where a regression hides.

### Replay against a candidate

```python
from shadoweval import Candidate, DivergenceDetector, Replayer, TokenOverlapScorer

candidate = Candidate(
    name="concise-v2",
    prompt_template="Answer the user briefly.\n\n{prompt}",
    version="2",
)

replayer = Replayer(
    TokenOverlapScorer(),
    detector=DivergenceDetector(threshold=0.6),
    candidate_cost_per_1k_tokens=0.003,
    candidate_latency_ms=910.0,
)
run = replayer.replay(sample, candidate)

print(len(run), run.divergence_rate, run.cost_usd)
```

### Decide, with statistics

```python
from shadoweval import RegressionAnalyzer

analyzer = RegressionAnalyzer(
    min_effect_size=0.02,  # ignore quality drops smaller than this
    alpha=0.05,
    min_samples=50,
    max_cost_ratio=1.25,  # block if the candidate costs >25% more
    max_latency_ratio=1.25,
)
verdict = analyzer.analyze(run.results)

print(verdict.decision.value)  # "ship" | "block" | "inconclusive"
print(verdict.p_value, verdict.effect_size, verdict.mean_delta)
for reason in verdict.reasons:
    print(" -", reason)
```

### Respect a budget

```python
from shadoweval import ReplayBudget

budget = ReplayBudget(max_cost_usd=0.01, max_requests=100, max_concurrency=4)
limited = replayer.replay(sample, candidate, budget=budget)

if limited.truncated:
    print("ran out of budget after", len(limited), "requests")
```

### Use the t-test directly

```python
from shadoweval import PairedTTest, student_t_pvalue

test = PairedTTest(alpha=0.05).run(run.baseline_scores, run.candidate_scores)
print(test.t_statistic, test.p_value, test.df, test.confidence_interval)

# The Student's t distribution is public API too.
print(student_t_pvalue(2.262, 9))  # 0.050012...
```

---

## 🧠 How it works

### Why a *paired* t-test

The baseline and the candidate answer the **same** production requests. Those
observations are not independent — pairing removes the variance caused by how
hard a particular question is, which is by far the largest source of noise.
The test works on the per-item differences `d_i = candidate_i - baseline_i`:

```
mean(d) = (1/n) * SUM d_i
std(d)  = sqrt( (1/(n-1)) * SUM (d_i - mean(d))^2 )
t       = mean(d) / ( std(d) / sqrt(n) )
```

with `n - 1` degrees of freedom. The p-value comes from the two-sided
Student's t distribution.

### The t distribution, implemented from scratch

`shadoweval.decision` evaluates the two-sided p-value through the regularised
incomplete beta function `I_x(a, b)`, using the identity

```
P(|T| > t) = I_{df/(df+t^2)} (df/2, 1/2)
```

`I_x` is computed with Lentz's modified continued fraction — numerically
stable for every `x` in `[0, 1]` — and the normalising constant comes from
`math.lgamma`:

```
I_x(a, b) = x^a (1-x)^b * Gamma(a+b) / (Gamma(a) Gamma(b)) * 1/a * CF(a, b, x)
```

The implementation is verified against published critical values to three
decimal places (see `tests/test_decision.py`):

| statistic | df | p-value |
|---|---|---|
| t = 2.000 | 10 | 0.073 |
| t = 2.000 | 9 | 0.077 |
| t = 2.262 | 9 | 0.050 |
| t = 1.833 | 9 | 0.100 |
| t = 3.250 | 9 | 0.010 |
| t = 4.781 | 9 | 0.001 |

### Effect size

A p-value tells you *whether* something changed; Cohen's *d* tells you
*how much*. For paired data the effect size is the mean difference over the
standard deviation of the differences:

```
d = mean(d) / std(d)
```

`min_effect_size` acts as a floor: a candidate that is significantly worse
but only by 0.001 is not worth blocking a release for.

### Bootstrap confidence interval

The interval around the mean score delta is estimated by a seeded bootstrap:
`resamples` (default 1000) resamples of the difference list are drawn with
replacement, and the 2.5% / 97.5% percentiles of their means form the
interval. It makes no normality assumption, it is deterministic, and its
width shrinks as the sample grows.

### The decision table

| Condition | Decision |
|---|---|
| `n < min_samples` | **INCONCLUSIVE** — not enough evidence |
| candidate significantly worse (`p < alpha`) **and** `mean_delta <= -min_effect_size` | **BLOCK** |
| candidate significantly worse (`p < alpha`) but drop below `min_effect_size` | **SHIP** (statistical noise) |
| candidate significantly better (`p < alpha`) | **SHIP** |
| no significant change (`p >= alpha`) **and** cost/latency within tolerance | **SHIP** |
| no significant change **but** cost ratio > `max_cost_ratio` | **BLOCK** (cost criterion) |
| no significant change **but** latency ratio > `max_latency_ratio` | **BLOCK** (latency criterion) |

### Divergence detection

`DivergenceDetector.similarity(a, b)` returns a number in `[0, 1]` from two
signals: the Jaccard index of the two answers' content tokens (0.8 weight)
and a numeric/statement check (0.2 weight). When both answers contain
numbers they must agree, and a negation conflict ("the account is active" vs
"the account is **not** active") subtracts a penalty. A reworded but
equivalent answer scores high; a different answer — or the same answer with
a different number — scores low.

### Sampling that does not waste money

`stratified` walks the strata round-robin so no tenant or tag is starved;
`error_focused` fills the sample with failures first; `recent` takes the
newest requests. Dedupe uses token Jaccard to drop near-duplicate prompts:
replaying fifty phrasings of the same question fifty times tells you nothing
about a prompt change.

---

## 📊 Example output

Run it yourself with `uv run python examples/quickstart.py`:

```text
==============================================================================
shadoweval - shadow traffic evaluation
==============================================================================
Captured 300 production requests across 3 tenants.

Sampled 100 requests (stratified by tenant, deduped).
  tenants: {'acme': 34, 'globex': 33, 'initech': 33}

Replayed 100 requests against 'concise-v2' v2.
  divergent answers : 99.0% of the sample
  candidate spend   : $0.0174

Verdict
------------------------------------------------------------------------------
  decision          : BLOCK
  sample size       : 100
  p-value           : 0.000000
  t-statistic       : -28.0900
  mean score delta  : -0.4394
  effect size (d)   : -2.8090
  95% CI (t)        : [-0.4704, -0.4083]
  95% CI (bootstrap): [-0.4745, -0.4115]
  cost delta        : $-0.000293 per request
  latency delta     : -184.0 ms per request
  reasons:
    - candidate scored significantly worse than baseline (p=5.815e-49 < 0.05, mean delta -0.4394)
    - quality drop 0.4394 exceeds the minimum effect size 0.0200

Budget-limited replay
------------------------------------------------------------------------------
  budget            : $0.01 / 100 requests
  requests replayed : 56 of 100
  truncated         : True
  spend             : $0.0097/$0.01 spent, 56/100 requests, concurrency<=4
  verdict (partial) : BLOCK (n=56, p=0.0000)

Control: redeploying the unchanged model
------------------------------------------------------------------------------
  decision          : SHIP
  p-value           : 1.000000
  mean score delta  : +0.0000
  effect size (d)   : +0.0000

Done.
```

The last block is the control: replaying the *unchanged* model against the
same traffic produces an identical score for every request, so the t-statistic
is exactly zero, the p-value is exactly 1.0, and the verdict is SHIP. A
harness that cannot do that is not measuring anything.

---

## 🧪 Testing
```bash
uv sync --all-extras --dev
uv run pytest -q --cov=shadoweval --cov-report=term-missing
uv run ruff check .
```

---

## 📄 License
MIT © 2026 Arun Tewatia
