"""Learning loop: refit scoring weights and narrative heat from outcomes.

Runs daily (or via --train). Three updates, all from the same labeled
decision set — buys AND skips, so the model also learns from what it
refused:

1. Weights: logistic regression on (features -> win/not-win), L2-anchored
   to the priors so small samples cannot fling the weights around. The
   anchor loosens as data accumulates.
2. Narrative heat: per-bucket smoothed win rate mapped to [-1, 1]. This is
   what 'adapting to the meta' means mechanically: buckets that win get
   positive heat and score higher tomorrow.
3. Age-bracket stats: fresh vs young win rates, reported and fed back the
   same way through the is_fresh weight.
"""

import json

import numpy as np

import config
import db
import narrative
import scoring

MIN_SAMPLES_TO_TRAIN = 40
L2_LAMBDA_BASE = 4.0     # strong anchor at small n, decays with sqrt(n)
LEARNING_RATE = 0.1
EPOCHS = 300


def _dataset(rows):
    X, y = [], []
    for row in rows:
        feats = json.loads(row["features"])
        if feats.get("_version") != scoring.FEATURE_VERSION:
            continue
        X.append([feats.get(name, 0.0) for name in scoring.FEATURE_NAMES])
        y.append(1.0 if row["label"] == "win" else 0.0)
    return np.array(X), np.array(y)


def fit_weights(rows):
    """Returns (weights dict, note) or (None, reason)."""
    X, y = _dataset(rows)
    n = len(y)
    if n < MIN_SAMPLES_TO_TRAIN:
        return None, f"only {n} labeled samples (< {MIN_SAMPLES_TO_TRAIN})"
    if y.sum() == 0 or y.sum() == n:
        return None, "labels are all one class; nothing separable yet"

    prior = np.array([scoring.PRIOR_WEIGHTS[name] for name in scoring.FEATURE_NAMES])
    w = prior.copy()
    lam = L2_LAMBDA_BASE / np.sqrt(n)

    # Never regularize the intercept: it must move freely to absorb the base
    # win rate. If it were anchored, the positive feature priors would inflate
    # the predicted log-odds and gradient descent would drag every weight down
    # to compensate — burying genuinely predictive features below their priors.
    reg_mask = np.array([0.0 if name == "bias" else 1.0
                         for name in scoring.FEATURE_NAMES])

    for _ in range(EPOCHS):
        p = 1.0 / (1.0 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) / n + lam * reg_mask * (w - prior)
        w -= LEARNING_RATE * grad

    weights = {name: round(float(v), 4)
               for name, v in zip(scoring.FEATURE_NAMES, w)}
    wins = int(y.sum())
    return weights, f"fit on {n} samples ({wins} wins), l2={lam:.3f}"


def bucket_stats(con):
    """Smoothed win rate per narrative bucket over recent labeled buys+skips."""
    rows = con.execute(
        "SELECT t.narrative bucket, o.label FROM decisions d"
        " JOIN outcomes o ON o.decision_id = d.id AND o.done = 1"
        " JOIN tokens t ON t.mint = d.mint"
        " WHERE o.label IS NOT NULL"
        " ORDER BY d.id DESC LIMIT 1000").fetchall()

    counts = {}
    for row in rows:
        b = row["bucket"] or "other"
        wins, total = counts.get(b, (0, 0))
        counts[b] = (wins + (row["label"] == "win"), total + 1)

    base_rate = 0.25  # prior win expectation; heat is deviation from this
    heat, stats = {}, {}
    for bucket, (wins, total) in counts.items():
        rate = (wins + 2 * base_rate) / (total + 2)   # Laplace-ish smoothing
        heat[bucket] = round(max(-1.0, min(1.0, (rate - base_rate) * 4)), 3)
        stats[bucket] = {"wins": wins, "total": total, "rate": round(rate, 3)}
    return heat, stats


def age_stats(con):
    rows = con.execute(
        "SELECT t.bracket, o.label, COUNT(*) n FROM decisions d"
        " JOIN outcomes o ON o.decision_id = d.id AND o.done = 1"
        " JOIN tokens t ON t.mint = d.mint"
        " WHERE o.label IS NOT NULL GROUP BY t.bracket, o.label").fetchall()
    out = {}
    for row in rows:
        b = out.setdefault(row["bracket"] or "unknown", {})
        b[row["label"]] = row["n"]
    return out


def run(con, log):
    """Full training pass. Returns the daily report text."""
    rows = db.labeled_decisions(con)
    weights, note = fit_weights(rows)
    if weights:
        version = db.save_weights(con, weights, note)
        log(f"[TRAIN] weights v{version}: {note}")
    else:
        log(f"[TRAIN] weights unchanged: {note}")

    heat, bstats = bucket_stats(con)
    db.set_meta(con, "narrative_heat", heat)
    astats = age_stats(con)
    db.set_meta(con, "age_stats", astats)

    import paper
    stats = {
        "portfolio": paper.portfolio_summary(con),
        "narrative_buckets": bstats,
        "narrative_heat": heat,
        "age_brackets": astats,
        "weights_note": note,
    }
    report = narrative.daily_report(stats)
    db.set_meta(con, "last_report", {"text": report})
    return report, stats
