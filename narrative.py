"""Meta layer: narrative classification and heat tracking.

The LLM does exactly two jobs, both advisory:
  1. classify a token name/symbol into a narrative bucket (cheap model)
  2. write the daily meta report from stats the trainer computed (smart model)
It never sees or sets prices, sizes or scores. Classification failures
fall back to 'other' with neutral heat — the pipeline never blocks on AI.

Heat itself is NOT set by the LLM: it is the smoothed win-rate of each
bucket from real outcomes, computed in trainer.py. The model only names
the bucket; the data decides whether that bucket is hot.
"""

import json

import requests

import config

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

CLASSIFY_PROMPT = (
    "Classify this Solana memecoin into exactly one bucket from: "
    + ", ".join(config.NARRATIVE_BUCKETS)
    + ". Respond with JSON only: {\"bucket\": \"...\"}. "
    "dog/cat/frog cover animal memes; ai covers AI-themed; brainrot covers "
    "absurdist internet-speak; degen covers gambling/casino themes; event "
    "covers holidays or current happenings; parody covers spoofs of known "
    "projects or brands; other when unsure."
)


def _chat(model, system, user, max_tokens=200, timeout=20):
    if not config.GROQ_API_KEY:
        return None
    try:
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {config.GROQ_API_KEY}"},
            json={
                "model": model,
                "temperature": 0.1,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        return json.loads(resp.json()["choices"][0]["message"]["content"])
    except (requests.RequestException, KeyError, IndexError,
            json.JSONDecodeError, ValueError):
        return None


def classify(name, symbol):
    """Returns a bucket name; 'other' on any failure."""
    verdict = _chat(config.GROQ_MODEL_FAST, CLASSIFY_PROMPT,
                    json.dumps({"name": name, "symbol": symbol}), max_tokens=50)
    bucket = (verdict or {}).get("bucket", "other")
    return bucket if bucket in config.NARRATIVE_BUCKETS else "other"


def heat_for(con_heat_map, bucket):
    """Heat in [-1, 1] for a bucket from the trainer-computed map."""
    return float((con_heat_map or {}).get(bucket, 0.0))


REPORT_PROMPT = (
    "You are the analyst for a memecoin paper-trading bot. You receive its "
    "recent performance statistics as JSON: per-narrative-bucket win rates, "
    "age-bracket results, notable wins/losses. Write a terse daily report "
    "(under 150 words): which narratives are hot or dying, whether fresh "
    "launches or young survivors are working better, and one concrete "
    "caution. Respond as JSON: {\"report\": \"...\"}."
)


def daily_report(stats):
    verdict = _chat(config.GROQ_MODEL_SMART, REPORT_PROMPT,
                    json.dumps(stats), max_tokens=400, timeout=30)
    return (verdict or {}).get("report", "(AI report unavailable)")
