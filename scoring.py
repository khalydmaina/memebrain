"""Judgment layer: feature extraction and the learned score.

Score = sigmoid(w . x) * 100. The weight vector starts from hand-set
priors and is re-fit by trainer.py from real outcomes. Feature extraction
is versioned: the trainer replays stored feature dicts, so changing a
feature's definition requires bumping FEATURE_VERSION and letting old
rows age out.
"""

import math

import config

FEATURE_VERSION = 1

FEATURE_NAMES = [
    "bias",
    "log_liquidity",      # log10(liq)/6 — $1M -> 1.0
    "vol_liq_ratio",      # 1h volume / liquidity, capped
    "momentum_m5",        # % / 25, clamped to [-1, 1]
    "momentum_h1",        # % / 100, clamped
    "buy_pressure",       # buys/(buys+sells) - 0.5, doubled
    "holders_scale",      # log10(holders)/4 — 10k holders -> 1.0
    "lp_locked",          # fraction of LP locked/burned
    "rugcheck_score",     # normalized 0..1 (higher = safer)
    "is_fresh",           # 1 if <6h old else 0
    "narrative_heat",     # learned per-bucket heat, [-1, 1]
]

# Priors: mild preference for traction, safety and hot narratives.
# The trainer overwrites these once real labels accumulate.
PRIOR_WEIGHTS = {
    "bias": -1.2,
    "log_liquidity": 0.6,
    "vol_liq_ratio": 0.8,
    "momentum_m5": 0.4,
    "momentum_h1": 0.5,
    "buy_pressure": 0.9,
    "holders_scale": 0.6,
    "lp_locked": 0.8,
    "rugcheck_score": 0.7,
    "is_fresh": 0.0,      # neutral prior: learning decides fresh vs young
    "narrative_heat": 0.8,
}


def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def extract_features(candidate, report, narrative_heat):
    liq = max(candidate.get("liquidity_usd") or 1, 1)
    buys = candidate.get("buys_h1") or 0
    sells = candidate.get("sells_h1") or 0
    total_tx = buys + sells
    holders = max(report["total_holders"], 1) if report else 1
    age = candidate.get("age_hours") or 0

    return {
        "_version": FEATURE_VERSION,
        "bias": 1.0,
        "log_liquidity": clamp(math.log10(liq) / 6, 0, 1),
        "vol_liq_ratio": clamp((candidate.get("vol_h1") or 0) / liq, 0, 1),
        "momentum_m5": clamp((candidate.get("chg_m5") or 0) / 25),
        "momentum_h1": clamp((candidate.get("chg_h1") or 0) / 100),
        "buy_pressure": clamp((buys / total_tx - 0.5) * 2) if total_tx else 0.0,
        "holders_scale": clamp(math.log10(holders) / 4, 0, 1),
        "lp_locked": clamp((report["lp_locked_pct"] if report else 0) / 100, 0, 1),
        "rugcheck_score": clamp((report["rugcheck_score"] if report else 0) / 100, 0, 1),
        "is_fresh": 1.0 if age < config.FRESH_AGE_HOURS else 0.0,
        "narrative_heat": clamp(narrative_heat),
    }


def score(features, weights=None):
    w = weights or PRIOR_WEIGHTS
    z = sum(w.get(name, 0.0) * features.get(name, 0.0) for name in FEATURE_NAMES)
    return 100.0 / (1.0 + math.exp(-z))


def explain(features, weights=None):
    """Top contributing features, for Telegram messages and logs."""
    w = weights or PRIOR_WEIGHTS
    contribs = [(name, w.get(name, 0.0) * features.get(name, 0.0))
                for name in FEATURE_NAMES if name != "bias"]
    contribs.sort(key=lambda t: abs(t[1]), reverse=True)
    return [f"{name} {'+' if v >= 0 else ''}{v:.2f}" for name, v in contribs[:4]]
