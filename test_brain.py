"""Unit tests: gates, scoring, paper exits, labeling, and the trainer."""

import json
import math
import os
import sqlite3
import tempfile

import numpy as np
import pytest

import config

# Point the DB at a temp file before importing db-dependent modules
config.DB_FILE = os.path.join(tempfile.mkdtemp(), "test.db")

import db
import paper
import redflags
import scoring
import trainer
from outcomes import _label


def good_candidate(**over):
    c = {"mint": "M" * 32, "pool": "P", "name": "DOG / SOL", "symbol": "DOG",
         "price_usd": 0.001, "liquidity_usd": 50000, "fdv_usd": 200000,
         "vol_h1": 20000, "vol_h24": 100000, "chg_m5": 5, "chg_h1": 30,
         "chg_h6": 50, "buys_h1": 300, "sells_h1": 200, "age_hours": 10}
    c.update(over)
    return c


def good_report(**over):
    r = {"mint_authority_active": False, "freeze_authority_active": False,
         "top10_pct": 20.0, "lp_locked_pct": 95.0, "total_holders": 800,
         "risks": [], "danger_risks": [], "rugcheck_score": 85.0}
    r.update(over)
    return r


# ── red flags ────────────────────────────────────────────────────────────────
def test_clean_token_passes():
    ok, flags = redflags.check(good_candidate(), good_report())
    assert ok, flags

def test_mint_authority_fails():
    ok, flags = redflags.check(good_candidate(), good_report(mint_authority_active=True))
    assert not ok and any("mint authority" in f for f in flags)

def test_no_report_fails_closed():
    ok, flags = redflags.check(good_candidate(), None)
    assert not ok and any("fail closed" in f for f in flags)

def test_honeypot_pattern_fails():
    ok, flags = redflags.check(good_candidate(buys_h1=50, sells_h1=0), good_report())
    assert not ok and any("honeypot" in f for f in flags)

def test_concentration_fails():
    ok, _ = redflags.check(good_candidate(), good_report(top10_pct=80))
    assert not ok

def test_thin_liquidity_fails():
    ok, _ = redflags.check(good_candidate(liquidity_usd=500), good_report())
    assert not ok


# ── scoring ──────────────────────────────────────────────────────────────────
def test_score_bounded_and_monotonic_in_heat():
    f_cold = scoring.extract_features(good_candidate(), good_report(), -1.0)
    f_hot = scoring.extract_features(good_candidate(), good_report(), 1.0)
    s_cold, s_hot = scoring.score(f_cold), scoring.score(f_hot)
    assert 0 <= s_cold <= 100 and 0 <= s_hot <= 100
    assert s_hot > s_cold

def test_better_token_scores_higher():
    weak = scoring.extract_features(
        good_candidate(liquidity_usd=11000, vol_h1=2500, buys_h1=10, sells_h1=30,
                       chg_h1=-20), good_report(rugcheck_score=40, lp_locked_pct=0), 0)
    strong = scoring.extract_features(good_candidate(), good_report(), 0.5)
    assert scoring.score(strong) > scoring.score(weak)


# ── outcome labeling ─────────────────────────────────────────────────────────
def test_label_win():
    assert _label({"rugged": 0, "m5": 1.1, "m30": 1.5, "m120": 1.2, "m1440": 0.9}) == "win"

def test_label_rug_is_loss():
    assert _label({"rugged": 1, "m5": 2.0, "m30": None, "m120": None, "m1440": None}) == "loss"

def test_label_flat():
    assert _label({"rugged": 0, "m5": 1.0, "m30": 1.1, "m120": 0.95, "m1440": 1.05}) == "flat"


# ── paper exits ──────────────────────────────────────────────────────────────
def test_stop_loss_closes_position():
    con = db.connect()
    db.open_position(con, "MINT1", "TST", entry_price=1.0, size_sol=0.25)
    events = paper.manage(con, {"MINT1": 0.65}, print)
    assert events and events[0][0] == "stop"
    pos = con.execute("SELECT * FROM positions WHERE mint='MINT1'").fetchone()
    assert pos["status"] == "closed" and pos["pnl_sol"] < 0

def test_tp_ladder():
    con = db.connect()
    db.open_position(con, "MINT2", "TST2", entry_price=1.0, size_sol=0.25)
    events = paper.manage(con, {"MINT2": 1.7}, print)   # +70% -> TP1
    assert events[0][0] == "tp1"
    events = paper.manage(con, {"MINT2": 2.6}, print)   # +160% -> TP2
    assert events[0][0] == "tp2"
    pos = con.execute("SELECT * FROM positions WHERE mint='MINT2'").fetchone()
    assert pos["status"] == "closed" and pos["pnl_sol"] > 0

def test_rug_closes_position():
    con = db.connect()
    db.open_position(con, "MINT3", "TST3", entry_price=1.0, size_sol=0.25)
    events = paper.manage(con, {"MINT3": 0.05}, print)
    assert events[0][0] == "rug"


# ── trainer ──────────────────────────────────────────────────────────────────
def _fake_rows(n, signal_feature="buy_pressure"):
    """Synthetic labeled decisions where one feature genuinely predicts wins."""
    rng = np.random.default_rng(7)
    rows = []
    for _ in range(n):
        feats = {name: float(rng.uniform(0, 1)) for name in scoring.FEATURE_NAMES}
        feats["bias"] = 1.0
        feats["_version"] = scoring.FEATURE_VERSION
        p_win = 0.15 + 0.6 * feats[signal_feature]
        label = "win" if rng.uniform() < p_win else "flat"
        rows.append({"features": json.dumps(feats), "action": "buy", "label": label})
    return rows

def test_trainer_needs_min_samples():
    weights, note = trainer.fit_weights(_fake_rows(10))
    assert weights is None and "samples" in note

def test_trainer_learns_the_signal():
    weights, note = trainer.fit_weights(_fake_rows(400))
    assert weights is not None
    # the predictive feature's weight should grow beyond its prior
    assert weights["buy_pressure"] > scoring.PRIOR_WEIGHTS["buy_pressure"]

def test_trainer_anchored_at_small_n():
    """With few samples the L2 anchor keeps weights near priors."""
    weights, _ = trainer.fit_weights(_fake_rows(45))
    drift = sum(abs(weights[k] - scoring.PRIOR_WEIGHTS[k]) for k in weights)
    weights_big, _ = trainer.fit_weights(_fake_rows(800))
    drift_big = sum(abs(weights_big[k] - scoring.PRIOR_WEIGHTS[k]) for k in weights_big)
    assert drift < drift_big


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
