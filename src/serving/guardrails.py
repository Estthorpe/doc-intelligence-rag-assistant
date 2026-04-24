# src/serving/guardrails.py
"""
Guardrail pipeline — applied before every /ask request.

Three layers in order:
1. Prompt injection detection (regex — fast, free, deterministic)
2. PII detection and redaction (Microsoft Presidio — local)
3. Off-topic blocking (keyword heuristics — fast, free)

All three run before any embedding, retrieval, or generation.
A violation returns immediately — no API calls are made.
Cost of a blocked request: $0.00.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

from src.config.logging_config import get_logger

logger = get_logger(__name__)


class ViolationType(str, Enum):
    PROMPT_INJECTION = "prompt_injection"
    PII_DETECTED = "pii_detected"
    OFF_TOPIC = "off_topic"


@dataclass
class GuardrailResult:
    """Result of running the guardrail pipeline."""

    passed: bool
    query: str  # original or redacted query
    violation_type: ViolationType | None = None
    violation_detail: str = ""
    pii_redacted: bool = False
    entities_found: list[str] = field(default_factory=list)


# ── Prompt injection patterns ─────────────────────────────────────────
INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+instructions",
    r"you\s+are\s+now\s+a",
    r"pretend\s+(you\s+are|to\s+be)",
    r"forget\s+(everything|all|your|the)",
    r"(reveal|show|output|print|display)\s+(your|the)\s+(system\s+prompt|instructions|training)",
    r"act\s+as\s+(a\s+|an\s+)?(different|new|unrestricted)",
    r"jailbreak",
    r"dan\s+mode",
    r"ignore\s+all\s+previous",
    r"new\s+persona",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# ── Off-topic keywords ────────────────────────────────────────────────
# Queries containing these are clearly not document Q&A requests
OFF_TOPIC_PATTERNS = [
    r"\bwrite\s+(me\s+)?(a\s+)?(poem|song|story|essay|joke|haiku)\b",
    r"\bwhat\s+is\s+the\s+weather\b",
    r"\bhelp\s+me\s+(code|program|debug|fix)\b",
    r"\btell\s+me\s+a\s+joke\b",
    r"\bwhat\s+is\s+\d+\s*[\+\-\*\/]\s*\d+\b",
    r"\btranslate\s+(this|to|from)\b",
    r"\bplay\s+(music|a\s+song)\b",
]
_OFF_TOPIC_RE = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_PATTERNS]

# ── PII entity types to detect ────────────────────────────────────────
PII_ENTITIES = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IBAN_CODE",
    "PERSON",
    "UK_NHS",
    "US_SSN",
    "US_BANK_NUMBER",
]

# Lazy-loaded Presidio engines
_analyzer: AnalyzerEngine | None = None
_anonymizer: AnonymizerEngine | None = None


def _get_presidio() -> tuple[AnalyzerEngine, AnonymizerEngine]:
    """Lazy-load Presidio engines on first use."""
    global _analyzer, _anonymizer
    if _analyzer is None:
        logger.info("Loading Presidio PII analyzer...")
        _analyzer = AnalyzerEngine()
        _anonymizer = AnonymizerEngine()
        logger.info("Presidio loaded")
    assert _anonymizer is not None
    return _analyzer, _anonymizer


class PromptInjectionGuard:
    """Detects prompt injection attempts via regex patterns."""

    def check(self, query: str) -> GuardrailResult:
        for pattern in _INJECTION_RE:
            if pattern.search(query):
                logger.warning(f"Prompt injection detected: pattern='{pattern.pattern[:40]}'")
                return GuardrailResult(
                    passed=False,
                    query=query,
                    violation_type=ViolationType.PROMPT_INJECTION,
                    violation_detail=(
                        "Your query contains patterns associated with "
                        "prompt injection attempts and cannot be processed."
                    ),
                )
        return GuardrailResult(passed=True, query=query)


class PIIGuard:
    """Detects and redacts PII using Microsoft Presidio."""

    def check(self, query: str) -> GuardrailResult:
        try:
            analyzer, anonymizer = _get_presidio()
            results = analyzer.analyze(
                text=query,
                entities=PII_ENTITIES,
                language="en",
            )

            if not results:
                return GuardrailResult(passed=True, query=query)

            entity_types = [r.entity_type for r in results]
            anonymized = anonymizer.anonymize(text=query, analyzer_results=results)
            redacted_query = anonymized.text

            logger.info(f"PII redacted from query: {entity_types}")

            return GuardrailResult(
                passed=True,  # Redact and continue — don't block
                query=redacted_query,
                pii_redacted=True,
                entities_found=entity_types,
                violation_detail=f"PII redacted: {', '.join(entity_types)}",
            )

        except Exception as e:
            logger.warning(f"PII check failed (non-blocking): {e}")
            return GuardrailResult(passed=True, query=query)


class OffTopicGuard:
    """Blocks queries that are clearly not document Q&A requests."""

    def check(self, query: str) -> GuardrailResult:
        for pattern in _OFF_TOPIC_RE:
            if pattern.search(query):
                logger.info(f"Off-topic query blocked: '{query[:60]}'")
                return GuardrailResult(
                    passed=False,
                    query=query,
                    violation_type=ViolationType.OFF_TOPIC,
                    violation_detail=(
                        "This assistant answers questions about uploaded "
                        "documents only. Please ask a question about the "
                        "content of your documents."
                    ),
                )
        return GuardrailResult(passed=True, query=query)


class GuardrailPipeline:
    """
    Runs all three guardrails in sequence.

    Order matters:
    1. Injection check first — cheapest, catches adversarial inputs
    2. PII redaction second — cleans the query before topic check
    3. Topic check last — operates on the cleaned query

    Returns on first violation — remaining checks are skipped.
    """

    def __init__(self) -> None:
        self.injection_guard = PromptInjectionGuard()
        self.pii_guard = PIIGuard()
        self.topic_guard = OffTopicGuard()

    def run(self, query: str) -> GuardrailResult:
        """
        Run all guardrails against the query.

        Args:
            query: The raw user query.

        Returns:
            GuardrailResult with passed=True and (possibly redacted)
            query if all checks pass, or passed=False with violation
            details if any check fails.
        """
        # Step 1: Injection check
        result = self.injection_guard.check(query)
        if not result.passed:
            return result

        # Step 2: PII redaction (modifies query, does not block)
        result = self.pii_guard.check(result.query)

        # Step 3: Topic check on (possibly redacted) query
        topic_result = self.topic_guard.check(result.query)
        if not topic_result.passed:
            return topic_result

        return result
