#!/usr/bin/env python3
"""Smoke tests for Groq inference with Llama 3.1 8B."""

import json
import os

import pytest
from openai import OpenAI

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


def _extract_json(response_text: str) -> dict:
    start = response_text.find("{")
    end = response_text.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("no JSON object found in response")
    return json.loads(response_text[start:end])


def _build_client(api_key: str) -> OpenAI:
    # Groq exposes an OpenAI-compatible API surface.
    return OpenAI(api_key=api_key, base_url=GROQ_BASE_URL)


def _run_basic_smoke_test(api_key: str) -> bool:
    client = _build_client(api_key)
    test_prompt = (
        "You are an ATC optimization expert. "
        "Return a JSON response with exactly these keys: model, task, status. "
        'Example: {"model": "llama-3.1-8b", "task": "atc_optimization", "status": "ready"}'
    )
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that responds with valid JSON only.",
            },
            {"role": "user", "content": test_prompt},
        ],
        temperature=0,
        max_tokens=200,
    )
    response_text = completion.choices[0].message.content or ""
    parsed = _extract_json(response_text)
    return {"model", "task", "status"}.issubset(parsed.keys())


def _run_atc_smoke_test(api_key: str) -> bool:
    client = _build_client(api_key)
    atc_prompt = """{
  "task": "delhi_monsoon_recovery_easy",
  "briefing": "Sample ATC runway schedule conflict scenario",
  "flights": 8,
  "heuristic_plan": [{"flight_id": "AI001", "runway": "09L", "assigned_minute": 5}]
}

You are optimizing an ATC runway schedule. Return strict JSON with keys: rationale, proposal.
proposal is an array of objects with flight_id, runway, assigned_minute, hold_minutes."""
    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a conservative ATC planner. Output strict JSON only.",
            },
            {"role": "user", "content": atc_prompt},
        ],
        temperature=0,
        max_tokens=500,
    )
    response_text = completion.choices[0].message.content or ""
    parsed = _extract_json(response_text)
    return "rationale" in parsed and "proposal" in parsed


def test_groq_inference() -> None:
    """Pytest entrypoint for a basic Groq JSON response."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GROQ_API_KEY is not set")
    assert _run_basic_smoke_test(api_key)


def test_atc_inference() -> None:
    """Pytest entrypoint for ATC-specific Groq JSON response."""
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        pytest.skip("GROQ_API_KEY is not set")
    assert _run_atc_smoke_test(api_key)


if __name__ == "__main__":
    print("=" * 70)
    print("GROQ INFERENCE TEST - Llama 3.1 8B")
    print("=" * 70)

    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        print("GROQ_API_KEY environment variable not set")
        print("Run: export GROQ_API_KEY='your-key-here'")
        print("\n" + "=" * 70)
        print("Some tests failed - check configuration above")
        print("=" * 70)
        raise SystemExit(0)

    success1 = _run_basic_smoke_test(api_key)
    success2 = _run_atc_smoke_test(api_key)

    print("\n" + "=" * 70)
    if success1 and success2:
        print("ALL TESTS PASSED - Groq inference is working")
        print("=" * 70)
    else:
        print("Some tests failed - check configuration above")
        print("=" * 70)
