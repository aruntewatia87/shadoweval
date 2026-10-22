"""Tests for traffic capture, sampling strategies and prompt dedupe."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import BASE_TIME, PROMPTS, make_request

from shadoweval import (
    ProductionRequest,
    SamplingStrategy,
    TrafficStore,
    dedupe,
    dedupe_prompts,
    is_error_request,
)
from shadoweval.text import extract_numbers, jaccard_similarity, tokenize


def test_store_records_and_counts_requests(store: TrafficStore) -> None:
    assert len(store) == 60
    recorded = store.get("req-0000")
    assert recorded is not None
    assert recorded.prompt == PROMPTS[0]
    assert store.get("does-not-exist") is None


def test_store_filter_by_tenant_and_tags() -> None:
    traffic = TrafficStore()
    traffic.record(make_request(0, tenant="acme", tags=("billing",)))
    traffic.record(make_request(1, tenant="globex", tags=("support",)))
    traffic.record(make_request(2, tenant="acme", tags=("billing", "support")))

    assert len(traffic.filter_by(tenant_id="acme")) == 2
    assert len(traffic.filter_by(tags=("billing",))) == 2
    assert len(traffic.filter_by(tags=("billing", "support"))) == 1
    assert len(traffic.filter_by(tenant_id="acme", tags=("support",))) == 1


def test_store_filter_by_feedback_and_time() -> None:
    traffic = TrafficStore()
    traffic.record(make_request(0, feedback=5.0, minutes_ago=0))
    traffic.record(make_request(1, feedback=1.0, minutes_ago=600))
    traffic.record(make_request(2, feedback=None, minutes_ago=1200))

    assert len(traffic.filter_by(min_feedback=4.0)) == 1
    assert len(traffic.filter_by(max_feedback=2.0)) == 1
    assert len(traffic.filter_by(has_feedback=True)) == 2
    assert len(traffic.filter_by(has_feedback=False)) == 1
    assert len(traffic.filter_by(since=BASE_TIME + timedelta(minutes=500))) == 2
    assert len(traffic.filter_by(until=BASE_TIME + timedelta(minutes=500))) == 1


def test_filter_by_prompt_contains() -> None:
    traffic = TrafficStore()
    traffic.record(make_request(0, prompt="refund my order"))
    traffic.record(make_request(1, prompt="reset my password"))
    assert len(traffic.filter_by(prompt_contains="REFUND")) == 1


def test_uniform_sampling_returns_requested_count(store: TrafficStore) -> None:
    sample = store.sample(20, seed=7)
    assert len(sample) == 20
    assert len({request.request_id for request in sample}) == 20


def test_sampling_is_deterministic_for_a_given_seed(store: TrafficStore) -> None:
    first = [r.request_id for r in store.sample(15, seed=42)]
    second = [r.request_id for r in store.sample(15, seed=42)]
    assert first == second


def test_different_seeds_produce_different_samples(store: TrafficStore) -> None:
    first = {r.request_id for r in store.sample(15, seed=1)}
    second = {r.request_id for r in store.sample(15, seed=2)}
    assert first != second


def test_sampling_larger_than_store_returns_everything(store: TrafficStore) -> None:
    assert len(store.sample(1000, seed=1)) == 60


def test_sampling_zero_returns_empty(store: TrafficStore) -> None:
    assert store.sample(0, seed=1) == []


def test_negative_sample_size_is_rejected(store: TrafficStore) -> None:
    with pytest.raises(ValueError, match="n must be non-negative"):
        store.sample(-1, seed=1)


def test_recent_sampling_returns_the_latest_requests(store: TrafficStore) -> None:
    sample = store.sample(5, seed=3, strategy=SamplingStrategy.RECENT)
    timestamps = [request.timestamp for request in sample]
    assert timestamps == sorted(timestamps, reverse=True)
    assert sample[0].request_id == "req-0059"


def test_stratified_sampling_balances_tenants() -> None:
    traffic = TrafficStore()
    for index in range(30):
        tenant = ("acme", "globex", "initech")[index % 3]
        traffic.record(make_request(index, tenant=tenant))
    sample = traffic.sample(30, seed=5, strategy=SamplingStrategy.STRATIFIED)
    counts: dict[str, int] = {}
    for request in sample:
        counts[request.tenant_id] = counts.get(request.tenant_id, 0) + 1
    assert counts == {"acme": 10, "globex": 10, "initech": 10}


def test_stratified_sampling_balances_uneven_tenants() -> None:
    traffic = TrafficStore()
    for index in range(21):
        tenant = "acme" if index < 20 else "globex"
        traffic.record(make_request(index, tenant=tenant))
    sample = traffic.sample(20, seed=5, strategy=SamplingStrategy.STRATIFIED)
    counts: dict[str, int] = {}
    for request in sample:
        counts[request.tenant_id] = counts.get(request.tenant_id, 0) + 1
    # The small stratum is exhausted first; the rest is filled from the large one.
    assert counts["globex"] == 1
    assert counts["acme"] == 19


def test_error_focused_sampling_prefers_failures() -> None:
    traffic = TrafficStore()
    for index in range(20):
        traffic.record(make_request(index))
    for index in range(20, 24):
        traffic.record(make_request(index, tags=("error",)))
    sample = traffic.sample(8, seed=11, strategy=SamplingStrategy.ERROR_FOCUSED)
    assert len(sample) == 8
    errored = [request for request in sample if request.is_error]
    assert len(errored) == 4
    # Every failure is picked up before any healthy request.
    assert sample[:4] == errored


def test_error_focused_sampling_falls_back_to_healthy_traffic() -> None:
    traffic = TrafficStore()
    for index in range(5):
        traffic.record(make_request(index))
    sample = traffic.sample(4, seed=11, strategy=SamplingStrategy.ERROR_FOCUSED)
    assert len(sample) == 4
    assert not any(request.is_error for request in sample)


def test_is_error_request_detects_feedback_and_tags() -> None:
    assert is_error_request(make_request(0, tags=("timeout",)))
    assert is_error_request(make_request(0, feedback=1.0))
    assert is_error_request(make_request(0, response=""))
    assert is_error_request(make_request(0, response="I cannot help with that"))
    assert not is_error_request(make_request(0, feedback=5.0))
    assert not is_error_request(make_request(0))


def test_dedupe_removes_near_duplicate_prompts() -> None:
    traffic = TrafficStore()
    traffic.record(make_request(0, prompt="how do I reset my password please"))
    traffic.record(make_request(1, prompt="how do I reset my password again"))
    traffic.record(make_request(2, prompt="where can I download my invoice"))
    # Token Jaccard of the first two prompts is 4/6 = 0.667.
    kept = dedupe(traffic.all(), similarity_threshold=0.6)
    assert len(kept) == 2
    assert kept[0].prompt.startswith("how do I reset")
    assert kept[1].prompt.startswith("where can I download")


def test_dedupe_keeps_distinct_prompts() -> None:
    traffic = TrafficStore()
    for index, prompt in enumerate(PROMPTS[:5]):
        traffic.record(make_request(index, prompt=prompt))
    kept = dedupe(traffic.all(), similarity_threshold=0.9)
    assert len(kept) == 5


def test_dedupe_keeps_everything_above_a_strict_threshold() -> None:
    traffic = TrafficStore()
    traffic.record(make_request(0, prompt="reset my password"))
    traffic.record(make_request(1, prompt="reset my password now"))
    # Token Jaccard is 3/4 = 0.75, below the strict 0.9 threshold.
    assert len(dedupe(traffic.all(), similarity_threshold=0.9)) == 2


def test_dedupe_always_removes_identical_prompts() -> None:
    traffic = TrafficStore()
    traffic.record(make_request(0, prompt="same prompt"))
    traffic.record(make_request(1, prompt="same prompt"))
    assert len(dedupe(traffic.all(), similarity_threshold=1.0)) == 1


def test_dedupe_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        dedupe([], similarity_threshold=1.5)


def test_dedupe_prompts_helper() -> None:
    prompts = ["reset my password", "reset my password now", "cancel my plan"]
    # "reset my password" vs "reset my password now" -> 3/4 = 0.75 token Jaccard.
    assert dedupe_prompts(prompts, similarity_threshold=0.7) == [
        "reset my password",
        "cancel my plan",
    ]


def test_sample_with_dedupe_threshold_shrinks_the_pool() -> None:
    traffic = TrafficStore()
    for index in range(6):
        traffic.record(make_request(index, prompt="identical prompt here"))
    sample = traffic.sample(10, seed=1, dedupe_threshold=0.8)
    assert len(sample) == 1


def test_jaccard_similarity_edge_cases() -> None:
    assert jaccard_similarity("", "") == 1.0
    assert jaccard_similarity("a b", "c d") == 0.0
    assert jaccard_similarity("a b", "a b") == 1.0
    assert 0.0 < jaccard_similarity("a b c", "a b d") < 1.0


def test_tokenize_strips_punctuation_and_case() -> None:
    assert tokenize("Hello, World! It's 42.") == ["hello", "world", "it's", "42"]


def test_extract_numbers_normalises_separators_and_percentages() -> None:
    assert extract_numbers("costs 1,234 dollars") == {"1234"}
    assert extract_numbers("up 25%") == {"25"}
    assert extract_numbers("no digits here") == set()
    assert extract_numbers("about 3.5 hours") == {"3.5"}


def test_production_request_helpers() -> None:
    request = ProductionRequest(
        request_id="r1",
        timestamp=BASE_TIME,
        prompt="hi",
        model="m",
        response="hello",
        latency_ms=100.0,
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.01,
    )
    assert request.total_tokens == 30
    assert not request.is_error


def test_store_extend_appends_many_requests() -> None:
    traffic = TrafficStore()
    traffic.extend(make_request(index) for index in range(5))
    assert len(traffic) == 5


def test_stratified_sampling_by_model() -> None:
    traffic = TrafficStore()
    for index in range(10):
        request = make_request(index)
        request.model = "model-a" if index % 2 == 0 else "model-b"
        traffic.record(request)
    sample = traffic.sample(8, seed=1, strategy=SamplingStrategy.STRATIFIED, strata_by="model")
    counts: dict[str, int] = {}
    for request in sample:
        counts[request.model] = counts.get(request.model, 0) + 1
    assert counts == {"model-a": 4, "model-b": 4}


def test_stratified_sampling_by_tag() -> None:
    traffic = TrafficStore()
    for index in range(10):
        tags = ("billing",) if index % 2 == 0 else ()
        traffic.record(make_request(index, tags=tags))
    sample = traffic.sample(
        8, seed=1, strategy=SamplingStrategy.STRATIFIED, strata_by="tag:billing"
    )
    tagged = [request for request in sample if request.tags]
    assert len(tagged) == 4


def test_stratified_sampling_by_feedback_bucket() -> None:
    traffic = TrafficStore()
    for index in range(10):
        feedback = 5.0 if index % 2 == 0 else None
        traffic.record(make_request(index, feedback=feedback))
    sample = traffic.sample(
        8, seed=1, strategy=SamplingStrategy.STRATIFIED, strata_by="feedback:low"
    )
    assert len(sample) == 8


def test_stratified_sampling_by_an_unknown_attribute() -> None:
    traffic = TrafficStore()
    for index in range(10):
        traffic.record(make_request(index))
    sample = traffic.sample(
        6, seed=1, strategy=SamplingStrategy.STRATIFIED, strata_by="not_a_field"
    )
    assert len(sample) == 6


def test_dedupe_prompts_rejects_an_invalid_threshold() -> None:
    with pytest.raises(ValueError, match="similarity_threshold"):
        dedupe_prompts(["a"], similarity_threshold=-0.5)


def test_prompt_fingerprint_is_stable_and_distinguishing() -> None:
    from shadoweval.capture import prompt_fingerprint

    assert prompt_fingerprint("hello") == prompt_fingerprint("hello")
    assert prompt_fingerprint("hello") != prompt_fingerprint("world")
