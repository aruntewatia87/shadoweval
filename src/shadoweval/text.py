"""Deterministic text utilities shared by the capture, scoring and replay layers.

Everything in this module is pure, offline and reproducible: no network access,
no model calls and no randomness.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?%?")

STOPWORDS: frozenset[str] = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    ]
)

_NEGATIONS: frozenset[str] = frozenset({"not", "no", "never", "none", "nor", "without", "cannot"})


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents and split ``text`` into word tokens."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    ascii_text = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return _TOKEN_RE.findall(ascii_text)


def content_tokens(text: str) -> list[str]:
    """Tokens of ``text`` with stopwords removed.

    Falls back to every token when the text contains nothing but stopwords so
    that short answers never collapse to an empty bag of words.
    """
    tokens = tokenize(text)
    meaningful = [token for token in tokens if token not in STOPWORDS]
    return meaningful or tokens


def _as_token_set(value: str | Iterable[str]) -> set[str]:
    """Coerce a string or an iterable of tokens into a token set."""
    if isinstance(value, str):
        return set(tokenize(value))
    return {str(token) for token in value}


def jaccard_similarity(a: str | Iterable[str], b: str | Iterable[str]) -> float:
    """Jaccard index (intersection over union) of two token sets.

    ``a`` and ``b`` may be raw strings or pre-tokenised iterables. Two empty
    sets count as identical (``1.0``); an empty set against a non-empty one
    scores ``0.0``.
    """
    set_a = _as_token_set(a)
    set_b = _as_token_set(b)
    union = set_a | set_b
    if not union:
        return 1.0
    return len(set_a & set_b) / len(union)


def extract_numbers(text: str) -> set[str]:
    """Extract normalised numeric literals from ``text``.

    Thousands separators are removed and a trailing ``%`` is dropped so that
    ``"1,234"`` and ``"1234"`` compare equal, while ``"42%"`` still compares
    equal to ``"42"``.
    """
    numbers: set[str] = set()
    for match in _NUMBER_RE.findall(text):
        cleaned = match.replace(",", "").rstrip("%")
        if not cleaned or cleaned in {"-", "+", "."}:
            continue
        try:
            value = float(cleaned)
        except ValueError:  # pragma: no cover - defensive, the regex is strict
            continue
        numbers.add(str(int(value)) if value.is_integer() else str(value))
    return numbers


def has_negation(text: str) -> bool:
    """Whether ``text`` contains a negation token such as ``not`` or ``don't``."""
    return any(token in _NEGATIONS or token.endswith("n't") for token in tokenize(text))
