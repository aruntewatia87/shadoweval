"""Shared fixtures and helpers for the shadoweval test suite."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from shadoweval import ProductionRequest, TrafficStore

BASE_TIME = datetime(2026, 3, 1, 9, 0, 0)

PROMPTS = [
    "How do I reset my password on the mobile app?",
    "Where can I download my invoice for last month?",
    "Why did my credit card payment fail?",
    "How do I change the email address on my account?",
    "What is the status of my refund request?",
    "How do I cancel my subscription before the renewal?",
    "Can I export my data to a CSV file?",
    "How do I add a teammate to my workspace?",
    "Why is my API key not working?",
    "How do I upgrade to the enterprise plan?",
]


def make_request(
    index: int,
    *,
    tenant: str = "acme",
    prompt: str | None = None,
    response: str | None = None,
    feedback: float | None = None,
    tags: tuple[str, ...] = (),
    minutes_ago: int | None = None,
) -> ProductionRequest:
    """Build a deterministic :class:`ProductionRequest` for tests."""
    text = prompt if prompt is not None else PROMPTS[index % len(PROMPTS)]
    first_word = text.split()[0].lower() if text.split() else "empty"
    answer = response if response is not None else f"Answer to {first_word} request number {index}."
    offset = minutes_ago if minutes_ago is not None else index
    return ProductionRequest(
        request_id=f"req-{index:04d}",
        timestamp=BASE_TIME + timedelta(minutes=offset),
        prompt=text,
        model="prod-model-v1",
        response=answer,
        latency_ms=800.0 + 10.0 * (index % 20),
        input_tokens=40 + (index % 15),
        output_tokens=60 + (index % 30),
        cost_usd=0.001 + 0.0001 * (index % 10),
        tenant_id=tenant,
        tags=tags,
        user_feedback=feedback,
    )


@pytest.fixture()
def store() -> TrafficStore:
    """A store with 60 requests spread over three tenants."""
    traffic = TrafficStore()
    for index in range(60):
        tenant = ("acme", "globex", "initech")[index % 3]
        traffic.record(make_request(index, tenant=tenant))
    return traffic
