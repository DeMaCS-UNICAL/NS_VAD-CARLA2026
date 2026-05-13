from __future__ import annotations

import json

from .types import Anomaly, CandidateAnomaly, ReasoningResult


def get_facts(line: str) -> list[str]:
    answer = _get_answer(line)
    facts: list[str] = []
    for part in answer.split("),"):
        fact = part.strip()
        if not fact:
            continue
        if not fact.endswith(")"):
            fact = f"{fact})"
        facts.append(fact)
    return facts


def get_anomalies(line: str) -> list[Anomaly]:
    return [
        Anomaly.from_string(fact)
        for fact in get_facts(line)
        if fact.startswith("anomaly(")
    ]


def get_candidate_anomalies(line: str) -> list[CandidateAnomaly]:
    return [
        CandidateAnomaly.from_string(fact)
        for fact in get_facts(line)
        if fact.startswith("candidate_anomaly(")
    ]


def parse_reasoning_result(line: str) -> ReasoningResult:
    facts = get_facts(line)
    return ReasoningResult(
        anomalies=[
            Anomaly.from_string(fact)
            for fact in facts
            if fact.startswith("anomaly(")
        ],
        candidate_anomalies=[
            CandidateAnomaly.from_string(fact)
            for fact in facts
            if fact.startswith("candidate_anomaly(")
        ],
    )


def _extract_json_or_raw_text(line: str) -> str:
    text = line.strip()
    json_start = text.find("{")
    if json_start >= 0:
        return text[json_start:].strip()
    return text


def _get_answer(line: str) -> str:
    text = _extract_json_or_raw_text(line)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text

    if not isinstance(payload, dict):
        raise ValueError(f"Expected DP-SR JSON object, got: {text}")

    answers = payload.get("answers")
    if (
        not isinstance(answers, list)
        or len(answers) != 1
        or not isinstance(answers[0], str)
    ):
        raise ValueError(
            "Expected DP-SR JSON output with exactly one string answer, "
            f"got: {text}"
        )

    return answers[0]
