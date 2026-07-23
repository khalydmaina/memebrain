"""Paper execution: simulated fills with realistic frictions.

Entries pay an estimated price impact based on pool depth plus a fixed
slippage allowance; exits pay it again. If paper results are good, live
results will be somewhat worse — never the other way round. That bias is
deliberate: the point of paper mode is to be hard to fool.
"""

import config
import db

FIXED_SLIPPAGE = 0.01  # 1% each way: priority fees, routing, timing


def price_impact(size_sol, liquidity_usd, sol_price_usd=200.0):
    """Constant-product approximation of our own impact on a thin pool."""
    size_usd = size_sol * sol_price_usd
    if liquidity_usd <= 0:
        return 0.10
    return min(size_usd / (liquidity_usd / 2), 0.10)


def buy(con, candidate):
    impact = price_impact(config.POSITION_SIZE_SOL, candidate["liquidity_usd"])
    fill = candidate["price_usd"] * (1 + impact + FIXED_SLIPPAGE)
    db.open_position(con, candidate["mint"], candidate["symbol"], fill,
                     config.POSITION_SIZE_SOL)
    return fill, impact


def _sell_value(pos, frac, price):
    """SOL received for selling `frac` of the original position at `price`."""
    mult = price / pos["entry_price"]
    gross = pos["size_sol"] * frac * mult
    return gross * (1 - FIXED_SLIPPAGE)


def manage(con, prices, log):
    """Walk open positions against current prices; apply stop/TP/time rules.
    Returns list of (event, position, detail) for notification."""
    events = []
    from datetime import datetime, timezone

    for pos in db.open_positions(con):
        price = prices.get(pos["mint"])
        if price is None:
            continue  # no fresh price this cycle; try next cycle

        change_pct = (price / pos["entry_price"] - 1) * 100
        opened = datetime.fromisoformat(pos["opened_ts"])
        held_hours = (datetime.now(timezone.utc) - opened).total_seconds() / 3600

        # Rug detection beats everything: mark and close at market
        if price <= pos["entry_price"] * config.RUG_THRESHOLD:
            pnl = _sell_value(pos, pos["remaining_frac"], price) \
                - pos["size_sol"] * pos["remaining_frac"]
            db.update_position(con, pos["id"], status="closed", exit_ts=db.now(),
                               exit_reason="rugged", pnl_sol=round(pnl, 4))
            events.append(("rug", pos, change_pct))
            continue

        if change_pct <= config.STOP_LOSS_PCT:
            pnl = _sell_value(pos, pos["remaining_frac"], price) \
                - pos["size_sol"] * pos["remaining_frac"]
            db.update_position(con, pos["id"], status="closed", exit_ts=db.now(),
                               exit_reason="stop_loss", pnl_sol=round(pnl, 4))
            events.append(("stop", pos, change_pct))
            continue

        if not pos["tp1_done"] and change_pct >= config.TP1_PCT:
            realized = _sell_value(pos, 0.5, price) - pos["size_sol"] * 0.5
            db.update_position(con, pos["id"], tp1_done=1, remaining_frac=0.5,
                               pnl_sol=round((pos["pnl_sol"] or 0) + realized, 4))
            events.append(("tp1", pos, change_pct))
            continue

        if pos["tp1_done"] and change_pct >= config.TP2_PCT:
            realized = _sell_value(pos, pos["remaining_frac"], price) \
                - pos["size_sol"] * pos["remaining_frac"]
            db.update_position(con, pos["id"], status="closed", exit_ts=db.now(),
                               exit_reason="tp2",
                               pnl_sol=round((pos["pnl_sol"] or 0) + realized, 4))
            events.append(("tp2", pos, change_pct))
            continue

        if held_hours >= config.MAX_HOLD_HOURS:
            pnl = _sell_value(pos, pos["remaining_frac"], price) \
                - pos["size_sol"] * pos["remaining_frac"]
            db.update_position(con, pos["id"], status="closed", exit_ts=db.now(),
                               exit_reason="time_stop",
                               pnl_sol=round((pos["pnl_sol"] or 0) + pnl, 4))
            events.append(("time", pos, change_pct))

    return events


def portfolio_summary(con):
    closed = con.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(pnl_sol),0) pnl,"
        " SUM(CASE WHEN pnl_sol > 0 THEN 1 ELSE 0 END) wins"
        " FROM positions WHERE status='closed'").fetchone()
    open_n = con.execute(
        "SELECT COUNT(*) n FROM positions WHERE status='open'").fetchone()["n"]
    return {
        "closed_trades": closed["n"],
        "wins": closed["wins"] or 0,
        "realized_pnl_sol": round(closed["pnl"], 4),
        "open_positions": open_n,
        "paper_balance_sol": round(config.PAPER_START_SOL + closed["pnl"], 4),
    }
