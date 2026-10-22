#!/usr/bin/env python3
"""End-to-end shadow evaluation demo.

Simulates a production LLM service, samples its traffic, replays it against a
candidate prompt that is subtly worse, and decides whether the candidate may
ship. The last section shows the same run under a tight budget.

Everything is deterministic and offline - no network, no API keys.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

from shadoweval import (
    Candidate,
    Decision,
    DivergenceDetector,
    PassthroughResponseGenerator,
    RegressionAnalyzer,
    ReplayBudget,
    Replayer,
    SamplingStrategy,
    TokenOverlapScorer,
    TrafficStore,
)
from shadoweval.capture import ProductionRequest
from shadoweval.text import content_tokens

BASE_TIME = datetime(2026, 3, 1, 9, 0, 0)

# The support questions the fictional service actually receives.
QUESTIONS = [
    "Where can I download my invoice for last month?",
    "How do I download a receipt for my subscription?",
    "Why is my credit card payment failing?",
    "Can I pay with a different credit card?",
    "How do I update my billing address?",
    "When will my card be charged next month?",
    "How do I get a VAT invoice with my company name?",
    "Can I switch to annual billing and get a refund?",
    "How do I remove an old expired card?",
    "Why was I charged twice this month?",
    "Do you offer refunds for partial months?",
    "How do I change the currency of my invoices?",
    "How do I reset my password on the mobile app?",
    "How do I change the email address on my account?",
    "How do I enable two factor authentication?",
    "How do I log out of all devices?",
    "Why did I get locked out of my account?",
    "How do I delete my account permanently?",
    "Can I have multiple logins for the same account?",
    "How do I recover access to my old email address?",
    "How do I change my username?",
    "What happens to my data if I close my account?",
    "How do I add a profile picture?",
    "How do I set up single sign on with Okta?",
    "How do I add a teammate to my workspace?",
    "How do I change a teammate role from viewer to editor?",
    "How do I remove a member from the workspace?",
    "Why did my teammate lose access after the trial ended?",
    "How many seats do I have on my plan?",
    "How do I buy more seats?",
    "Can a guest access the workspace?",
    "How do I transfer workspace ownership?",
    "What permissions does the auditor role have?",
    "How do I invite a team from a domain?",
    "How do I set a default role for new members?",
    "How do I see who is currently online?",
    "Why is my API key not working?",
    "Where do I find my API key?",
    "How do I rotate my API key without downtime?",
    "Does the webhook retry failed deliveries?",
    "How do I verify a webhook signature?",
    "Why am I getting a rate limit error?",
    "How do I increase my rate limit?",
    "Which SDK versions are still supported?",
    "How do I connect the Salesforce integration?",
    "Why did the Slack integration stop posting messages?",
    "How do I send events to the S3 sink?",
    "What happens if the integration times out?",
    "How do I test the integration in a sandbox?",
    "Can I filter which events reach the integration?",
    "Can I export my data to a CSV file?",
    "Where can I download my data archive?",
    "How long is my data kept after deletion?",
    "Can I export the audit log?",
    "How do I delete a single record permanently?",
    "Is my data encrypted at rest?",
    "Which data residency regions are available?",
    "How do I move my workspace to another region?",
    "Can I schedule a weekly data export?",
    "What is the retention limit on the event stream?",
    "How do I back up my workspace data?",
    "Does the export include attachments?",
    "How do I cancel my subscription before the renewal?",
    "How do I upgrade to the enterprise plan?",
    "What is the difference between the pro and business plans?",
    "How do I apply a promo code?",
    "Why was my discount removed after renewal?",
    "Can I pause my subscription for a month?",
    "How do I reactivate a cancelled subscription?",
    "Do you offer a student discount?",
    "How much does the extra usage cost?",
    "Can I set a spending limit for overage?",
    "How do I downgrade without losing data?",
    "What happens when my trial ends?",
    "The page keeps loading forever, what should I try?",
    "Why am I seeing a blank dashboard?",
    "What does the red status banner mean?",
    "How do I clear my browser cache for the app?",
    "Why is the mobile app crashing on launch?",
    "The report shows stale data, how do I refresh it?",
    "Why is search not returning any results?",
    "What does error code E-104 mean?",
    "How do I contact support outside business hours?",
    "Can I share a screen recording with support?",
    "Why do I get a blank email notification?",
    "The CSV download stops at eighty percent, what now?",
    "How do I change the notification frequency?",
    "Where can I see my usage this billing period?",
    "How do I set up budget alerts?",
    "Can I exclude a metric from the usage report?",
    "How do I grant read only access to a contractor?",
    "Why did my scheduled job not run?",
    "How do I retry a failed job?",
    "Can I run a job manually?",
    "Where do I find the job history?",
    "How do I pin a job to a specific worker?",
    "What is the default timeout for a job?",
    "How do I increase the worker memory limit?",
    "Can I see which worker processed my request?",
    "How do I drain a worker before deploying?",
]

TENANTS = ("acme", "globex", "initech")

# The canned "how to" paragraphs the support model draws on.
ANSWER_LEADS = (
    "Open Settings, choose Account and select Reset password. A reset link is "
    "emailed to you within two minutes.",
    "Your invoice is available under Settings and Billing. Every invoice can be "
    "downloaded as a PDF for the last 24 months.",
    "A payment can fail when the card has expired or the issuer declines the charge. "
    "Check the billing date and try a different payment method.",
    "You can change the email address under Settings, then Profile, then Email. "
    "A confirmation link is sent to the new address.",
    "Your refund was processed five business days ago and will appear on your next "
    "statement. Refunds usually settle within ten business days.",
    "You can cancel your subscription under Settings, then Billing, then Cancel plan. "
    "The plan stays active until the end of the billing period.",
    "Go to Settings, then Data, then Export. The archive is emailed as a zip file within an hour.",
    "Invite teammates from Settings, then Members, then Invite. Enter their email "
    "address and choose a role.",
    "An API key works only if it is active and the account is in good standing. "
    "Keys are scoped to the workspace that created them.",
    "The enterprise plan includes SSO, audit logs and a 99.9 percent uptime SLA. "
    "Contact sales for a tailored quote.",
    "The export contains every record in the workspace as one file per entity, "
    "including soft deleted entries from the last ninety days.",
    "Run the job from the Jobs page and pick Retry. Failed attempts keep their logs "
    "so you can inspect the exact error before rerunning.",
)


def stable_index(text: str, modulo: int) -> int:
    """Pick a stable bucket for ``text``.

    Python's built-in :func:`hash` is salted per process, so a reproducible
    demo has to use a cryptographic hash instead.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % modulo


def answer_for(question: str, quality: float) -> str:
    """Render a deterministic support answer for ``question``.

    ``quality`` scales how much of the answer the model actually produces: a
    model at 1.0 writes the full paragraph, a model at 0.45 answers in one
    breath. The production traffic is recorded at 1.0, the candidate model
    answers at 0.45 - which is exactly the "subtly worse" prompt change this
    demo evaluates.
    """
    lead = ANSWER_LEADS[stable_index(question, len(ANSWER_LEADS))]
    lead_terms = list(dict.fromkeys(content_tokens(lead)))
    question_terms = list(dict.fromkeys(content_tokens(question)))
    keep_lead = max(3, round(len(lead_terms) * quality))
    keep_question = max(1, round(len(question_terms) * quality))
    return " ".join(lead_terms[:keep_lead] + question_terms[:keep_question])


@dataclass(slots=True)
class SupportModel:
    """A deterministic offline stand-in for the candidate model.

    It implements the :class:`~shadoweval.replay.ResponseGenerator` protocol:
    given the rendered prompt and the original request it returns the answer
    the candidate model would give.
    """

    quality: float = 0.45

    def generate(self, prompt: str, request: ProductionRequest) -> str:
        """Return the candidate model's answer to ``request``."""
        return answer_for(request.prompt, self.quality)


def build_traffic(count: int = 300) -> TrafficStore:
    """Build a deterministic store of production traffic.

    A handful of the requests are failures: timeouts, API errors and one or
    two badly rated answers. Regressions hide in exactly those requests, so
    the store has to contain them.
    """
    store = TrafficStore()
    for index in range(count):
        question = QUESTIONS[index % len(QUESTIONS)]
        answer = answer_for(question, 1.0)
        tenant = TENANTS[index % len(TENANTS)]
        tags: tuple[str, ...] = ()
        feedback: float | None = None

        if index % 37 == 0:
            answer = "Error: upstream request timed out after 30 seconds."
            tags = ("error", "timeout")
        elif index % 53 == 0:
            answer = "I cannot help with that request."
            tags = ("error", "refusal")
        elif index % 29 == 0:
            feedback = 1.0
            answer = "Not sure, you should probably check the documentation."

        store.record(
            ProductionRequest(
                request_id=f"req-{index:05d}",
                timestamp=BASE_TIME + timedelta(minutes=index),
                prompt=question,
                model="support-v1",
                response=answer,
                latency_ms=780.0 + 40.0 * (index % 17),
                input_tokens=42 + (index % 21),
                output_tokens=88 + (index % 34),
                cost_usd=0.0012 + 0.0001 * (index % 7),
                tenant_id=tenant,
                tags=tags,
                user_feedback=feedback,
            )
        )
    return store


def main() -> None:
    """Run the demo."""
    store = build_traffic()
    print("=" * 78)
    print("shadoweval - shadow traffic evaluation")
    print("=" * 78)
    print(f"Captured {len(store)} production requests across {len(TENANTS)} tenants.")

    # ------------------------------------------------------------------
    # 1. Sample the traffic.
    # ------------------------------------------------------------------
    sample = store.sample(
        100,
        seed=20260301,
        strategy=SamplingStrategy.STRATIFIED,
        strata_by="tenant_id",
        dedupe_threshold=0.7,
    )
    tenants: dict[str, int] = {}
    for request in sample:
        tenants[request.tenant_id] = tenants.get(request.tenant_id, 0) + 1
    print(f"\nSampled {len(sample)} requests (stratified by tenant, deduped).")
    print(f"  tenants: {dict(sorted(tenants.items()))}")

    # ------------------------------------------------------------------
    # 2. Replay against a subtly worse candidate prompt.
    # ------------------------------------------------------------------
    candidate = Candidate(
        name="concise-v2",
        prompt_template="Answer the user briefly and get to the point.\n\n{prompt}",
        version="2",
    )
    replayer = Replayer(
        TokenOverlapScorer(),
        generator=SupportModel(quality=0.45),
        detector=DivergenceDetector(threshold=0.6),
        candidate_cost_per_1k_tokens=0.003,
        candidate_latency_ms=910.0,
    )
    run = replayer.replay(sample, candidate)

    print(f"\nReplayed {len(run)} requests against '{candidate.name}' v{candidate.version}.")
    print(f"  divergent answers : {run.divergence_rate:.1%} of the sample")
    print(f"  candidate spend   : ${run.cost_usd:.4f}")

    # ------------------------------------------------------------------
    # 3. Decide, with statistics.
    # ------------------------------------------------------------------
    analyzer = RegressionAnalyzer(
        min_effect_size=0.02,
        alpha=0.05,
        min_samples=50,
        max_cost_ratio=1.25,
        max_latency_ratio=1.25,
    )
    verdict = analyzer.analyze(run.results)

    print("\nVerdict")
    print("-" * 78)
    print(f"  decision          : {verdict.decision.value.upper()}")
    print(f"  sample size       : {verdict.sample_size}")
    print(f"  p-value           : {verdict.p_value:.6f}")
    print(f"  t-statistic       : {verdict.t_statistic:+.4f}")
    print(f"  mean score delta  : {verdict.mean_delta:+.4f}")
    print(f"  effect size (d)   : {verdict.effect_size:+.4f}")
    if verdict.confidence_interval is not None:
        interval = verdict.confidence_interval
        print(f"  95% CI (t)        : [{interval.lower:+.4f}, {interval.upper:+.4f}]")
    if verdict.bootstrap_interval is not None:
        interval = verdict.bootstrap_interval
        print(f"  95% CI (bootstrap): [{interval.lower:+.4f}, {interval.upper:+.4f}]")
    print(f"  cost delta        : ${verdict.cost_delta:+.6f} per request")
    print(f"  latency delta     : {verdict.latency_delta:+.1f} ms per request")
    print("  reasons:")
    for reason in verdict.reasons:
        print(f"    - {reason}")

    # ------------------------------------------------------------------
    # 4. The same run under a tight budget.
    # ------------------------------------------------------------------
    budget = ReplayBudget(max_cost_usd=0.01, max_requests=100, max_concurrency=4)
    limited = replayer.replay(sample, candidate, budget=budget)
    print("\nBudget-limited replay")
    print("-" * 78)
    print(f"  budget            : ${budget.max_cost_usd:.2f} / {budget.max_requests} requests")
    print(f"  requests replayed : {len(limited)} of {len(sample)}")
    print(f"  truncated         : {limited.truncated}")
    print(f"  spend             : {budget.summary()}")

    partial = analyzer.analyze(limited.results)
    print(
        f"  verdict (partial) : {partial.decision.value.upper()} "
        f"(n={partial.sample_size}, p={partial.p_value:.4f})"
    )

    # ------------------------------------------------------------------
    # 5. A safe candidate ships.
    # ------------------------------------------------------------------
    same = Candidate(name="support-v1-redeploy", prompt_template="{prompt}", version="1.1")
    safe = Replayer(TokenOverlapScorer(), generator=PassthroughResponseGenerator()).replay(
        sample, same
    )
    safe_verdict = analyzer.analyze(safe.results)
    print("\nControl: redeploying the unchanged model")
    print("-" * 78)
    print(f"  decision          : {safe_verdict.decision.value.upper()}")
    print(f"  p-value           : {safe_verdict.p_value:.6f}")
    print(f"  mean score delta  : {safe_verdict.mean_delta:+.4f}")
    print(f"  effect size (d)   : {safe_verdict.effect_size:+.4f}")

    assert verdict.decision is Decision.BLOCK
    assert safe_verdict.decision is Decision.SHIP
    print("\nDone.")


if __name__ == "__main__":
    main()
