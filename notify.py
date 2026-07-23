"""Telegram alerts with retry (a lost alert cannot be regenerated)."""

import html
import time

import requests

import config


def send(message, retries=3):
    if not config.TELEGRAM_BOT_TOKEN or not config.TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": config.TELEGRAM_CHAT_ID, "text": message,
               "parse_mode": "HTML", "disable_web_page_preview": True}
    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, json=payload, timeout=20)
            if resp.status_code == 200 and resp.json().get("ok"):
                return True
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                print(f"[TG] rejected: {resp.text[:120]}")
                return False
        except requests.RequestException as e:
            print(f"[TG] attempt {attempt}: {e}")
        if attempt < retries:
            time.sleep(5 * attempt)
    return False


def esc(s):
    return html.escape(str(s))


def _usd(v):
    return f"${v:,.0f}" if v else "n/a"


def paper_buy(candidate, fill, impact, score_val, reasons, bucket):
    return "\n".join([
        f"🧠 <b>PAPER BUY {esc(candidate['symbol'])}</b>  (score {score_val:.0f})",
        f"Narrative: {esc(bucket)} | Age: {candidate['age_hours']:.1f}h",
        f"Fill: ${fill:.8f} (impact {impact*100:.1f}%)",
        f"MC: {_usd(candidate.get('fdv_usd'))} | "
        f"Liquidity: ${candidate['liquidity_usd']:,.0f} | "
        f"1h vol: ${candidate['vol_h1']:,.0f}",
        f"Drivers: {esc(', '.join(reasons))}",
        f"<code>{esc(candidate['mint'])}</code>",
    ])


def paper_exit(event, pos, change_pct):
    tags = {"rug": "💀 RUGGED", "stop": "🛑 STOP", "tp1": "✅ TP1 (half out)",
            "tp2": "🎯 TP2 (closed)", "time": "⏰ TIME STOP"}
    return (f"{tags.get(event, event)} <b>{esc(pos['symbol'])}</b> "
            f"{change_pct:+.0f}% | entry ${pos['entry_price']:.8f}")


def daily(report, stats):
    p = stats["portfolio"]
    lines = [
        "📊 <b>memebrain daily report</b>",
        f"Balance: {p['paper_balance_sol']} SOL (paper) | "
        f"{p['closed_trades']} closed, {p['wins']} wins | "
        f"open: {p['open_positions']}",
        "",
        esc(report),
    ]
    heat = stats.get("narrative_heat") or {}
    if heat:
        hot = sorted(heat.items(), key=lambda kv: -kv[1])[:3]
        lines.append("")
        lines.append("Meta heat: " + ", ".join(f"{b} {h:+.2f}" for b, h in hot))
    return "\n".join(lines)
