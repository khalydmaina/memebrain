"""memebrain: a paper-trading memecoin engine that learns from outcomes.

Usage:
  python main.py --once             one scan cycle, then exit
  python main.py --loop             continuous scanning
  python main.py --train            run the learning pass now
  python main.py --status           portfolio + learning state
  python main.py --test-telegram    send a test alert
"""

import argparse
import sys
import time
from datetime import datetime, timezone

import config
import db
import narrative
import notify
import outcomes
import paper
import redflags
import scoring
import sources
import trainer


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        with open(config.LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def consider(con, candidate, weights, weights_version, heat_map):
    """Gate -> classify -> score -> decide. Every path records a decision."""
    mint = candidate["mint"]

    report = sources.rugcheck_report(mint)
    passed, flags = redflags.check(candidate, report)

    bucket = "other"
    if not db.token_known(con, mint):
        if passed:  # only spend LLM calls on tokens that cleared the gates
            bucket = narrative.classify(candidate["name"], candidate["symbol"])
        age = candidate.get("age_hours") or 0
        bracket = "fresh" if age < config.FRESH_AGE_HOURS else "young"
        db.save_token(con, mint, candidate["symbol"], candidate["name"],
                      age, bracket, bucket, candidate)
    else:
        row = con.execute("SELECT narrative FROM tokens WHERE mint=?",
                          (mint,)).fetchone()
        bucket = row["narrative"] if row else "other"

    feats = scoring.extract_features(candidate, report,
                                     narrative.heat_for(heat_map, bucket))

    if not passed:
        db.record_decision(con, mint, "skip", 0.0, candidate["price_usd"],
                           feats, weights_version, "; ".join(flags[:4]))
        log(f"  {candidate['symbol']:12} REJECT: {flags[0]}")
        return

    score_val = scoring.score(feats, weights)

    if score_val < config.MIN_BUY_SCORE:
        db.record_decision(con, mint, "skip", score_val, candidate["price_usd"],
                           feats, weights_version, f"score {score_val:.0f}")
        log(f"  {candidate['symbol']:12} pass gates, score {score_val:.0f} — skip")
        return

    if len(db.open_positions(con)) >= config.MAX_OPEN_POSITIONS:
        db.record_decision(con, mint, "skip", score_val, candidate["price_usd"],
                           feats, weights_version, "position limit")
        log(f"  {candidate['symbol']:12} score {score_val:.0f} but at position limit")
        return
    if db.positions_opened_today(con) >= config.MAX_NEW_POSITIONS_PER_DAY:
        db.record_decision(con, mint, "skip", score_val, candidate["price_usd"],
                           feats, weights_version, "daily limit")
        return

    fill, impact = paper.buy(con, candidate)
    db.record_decision(con, mint, "buy", score_val, candidate["price_usd"],
                       feats, weights_version, "bought (paper)")
    reasons = scoring.explain(feats, weights)
    log(f"  {candidate['symbol']:12} PAPER BUY @ ${fill:.8f} (score {score_val:.0f})")
    notify.send(notify.paper_buy(candidate, fill, impact, score_val, reasons, bucket))


def cycle(con):
    log("scan cycle start")
    weights_version, weights = db.latest_weights(con)
    heat_map = db.get_meta(con, "narrative_heat", {})

    candidates = sources.discover_pools(pages=config.GECKO_PAGES)
    log(f"discovered {len(candidates)} pools")

    fresh_mints = [c for c in candidates
                   if not con.execute(
                       "SELECT 1 FROM decisions WHERE mint=? AND ts > datetime('now','-6 hours')",
                       (c["mint"],)).fetchone()]
    log(f"{len(fresh_mints)} not decided on in the last 6h")

    for candidate in fresh_mints:
        if not candidate.get("price_usd"):
            continue
        try:
            consider(con, candidate, weights, weights_version, heat_map)
        except Exception as e:
            log(f"  {candidate.get('symbol','?')} error: {e}")

    # Manage open paper positions
    open_pos = db.open_positions(con)
    if open_pos:
        prices = sources.batch_prices([p["mint"] for p in open_pos])
        for event, pos, chg in paper.manage(con, prices, log):
            log(f"  position {pos['symbol']}: {event} at {chg:+.0f}%")
            notify.send(notify.paper_exit(event, pos, chg))

    # Sample outcome checkpoints
    n = outcomes.process(con, log)
    if n:
        log(f"sampled {n} outcome checkpoints")

    log("scan cycle end")


def maybe_train(con, last_trained_day):
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if last_trained_day == today:
        return last_trained_day
    report, stats = trainer.run(con, log)
    notify.send(notify.daily(report, stats))
    return today


def status(con):
    p = paper.portfolio_summary(con)
    version, weights = db.latest_weights(con)
    heat = db.get_meta(con, "narrative_heat", {})
    counts = con.execute(
        "SELECT action, COUNT(*) n FROM decisions GROUP BY action").fetchall()
    labeled = con.execute(
        "SELECT label, COUNT(*) n FROM outcomes WHERE done=1 GROUP BY label").fetchall()

    print("\n=== memebrain status ===")
    print(f"paper balance : {p['paper_balance_sol']} SOL "
          f"(started {config.PAPER_START_SOL})")
    print(f"closed trades : {p['closed_trades']} ({p['wins']} wins) | "
          f"open: {p['open_positions']}")
    print(f"decisions     : " + ", ".join(f"{r['action']}={r['n']}" for r in counts)
          if counts else "decisions     : none yet")
    print(f"labeled       : " + (", ".join(f"{r['label']}={r['n']}" for r in labeled)
                                 if labeled else "none settled yet"))
    print(f"weights       : v{version} " + ("(priors)" if not weights else "(learned)"))
    if heat:
        hot = sorted(heat.items(), key=lambda kv: -kv[1])
        print("meta heat     : " + ", ".join(f"{b}:{h:+.2f}" for b, h in hot[:6]))
    report = db.get_meta(con, "last_report")
    if report:
        print(f"\nlast AI report:\n{report['text']}\n")


def main():
    parser = argparse.ArgumentParser(description="memecoin paper trader that learns")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--test-telegram", action="store_true")
    args = parser.parse_args()

    con = db.connect()

    if args.test_telegram:
        config.validate_config()
        ok = notify.send("🧠 memebrain test message — delivery works.")
        print("sent" if ok else "FAILED")
        sys.exit(0 if ok else 1)
    if args.status:
        status(con)
        return
    if args.train:
        report, stats = trainer.run(con, log)
        notify.send(notify.daily(report, stats))
        print(f"\n{report}\n")
        return
    if not (args.once or args.loop):
        parser.print_help()
        return

    config.validate_config()
    log("memebrain starting (PAPER MODE — no real funds at risk)")
    log(f"gates: liq>=${config.MIN_LIQUIDITY_USD:,.0f}, "
        f"score>={config.MIN_BUY_SCORE}, top10<={config.MAX_TOP10_HOLDER_PCT}%")

    if args.once:
        cycle(con)
        return

    last_trained = None
    while True:
        try:
            cycle(con)
            last_trained = maybe_train(con, last_trained)
        except KeyboardInterrupt:
            log("stopped by user")
            break
        except Exception as e:
            log(f"cycle error: {e}")
        time.sleep(config.SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
