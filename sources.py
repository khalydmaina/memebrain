"""Market data: GeckoTerminal for discovery, DexScreener for prices, RugCheck
for on-chain safety. All keyless. Every function degrades to None/[] on
failure — the caller decides whether that fails a trade closed."""

import time
from datetime import datetime, timezone

import requests

GECKO = "https://api.geckoterminal.com/api/v2"
DEXSCREENER = "https://api.dexscreener.com"
RUGCHECK = "https://api.rugcheck.xyz/v1"

_HEADERS = {"User-Agent": "memebrain/1.0", "Accept": "application/json"}
_last_call = {}
# Per-host pacing: GeckoTerminal free tier allows 30 calls/min
_SPACING = {"api.geckoterminal.com": 2.1, "api.rugcheck.xyz": 1.5,
            "api.dexscreener.com": 0.25}


def _get(url, timeout=15):
    host = url.split("/")[2]
    wait = _SPACING.get(host, 0.5) - (time.monotonic() - _last_call.get(host, 0))
    if wait > 0:
        time.sleep(wait)
    _last_call[host] = time.monotonic()
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code == 429:
            time.sleep(20)
            resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        if resp.status_code != 200:
            return None
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_pool(item):
    """Normalize one GeckoTerminal pool into our candidate dict."""
    a = item.get("attributes", {})
    rel = item.get("relationships", {})
    base_id = rel.get("base_token", {}).get("data", {}).get("id", "")
    mint = base_id.split("_", 1)[1] if "_" in base_id else None
    if not mint:
        return None

    created = a.get("pool_created_at")
    age_hours = None
    if created:
        try:
            dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
            age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
        except ValueError:
            pass

    vol = a.get("volume_usd", {}) or {}
    chg = a.get("price_change_percentage", {}) or {}
    tx = (a.get("transactions", {}) or {}).get("h1", {}) or {}
    name = a.get("name", "")

    return {
        "mint": mint,
        "pool": a.get("address"),
        "name": name,
        "symbol": name.split(" / ")[0].strip() if " / " in name else name[:12],
        "price_usd": _f(a.get("base_token_price_usd")),
        "liquidity_usd": _f(a.get("reserve_in_usd")),
        "fdv_usd": _f(a.get("fdv_usd")),
        "vol_h1": _f(vol.get("h1")),
        "vol_h24": _f(vol.get("h24")),
        "chg_m5": _f(chg.get("m5")),
        "chg_h1": _f(chg.get("h1")),
        "chg_h6": _f(chg.get("h6")),
        "buys_h1": int(tx.get("buys") or 0),
        "sells_h1": int(tx.get("sells") or 0),
        "age_hours": age_hours,
    }


def discover_pools(pages=2):
    """New + trending Solana pools, deduped by mint. New pools are the
    'fresh' hunting ground; trending catches 'young survivors' gaining
    traction after launch."""
    seen, out = set(), []
    for endpoint in ("new_pools", "trending_pools"):
        for page in range(1, pages + 1):
            data = _get(f"{GECKO}/networks/solana/{endpoint}?page={page}")
            if not data:
                break
            for item in data.get("data", []):
                cand = _parse_pool(item)
                if cand and cand["mint"] not in seen:
                    seen.add(cand["mint"])
                    out.append(cand)
    return out


def batch_prices(mints):
    """Current USD prices for up to 30 mints per call via DexScreener."""
    prices = {}
    for i in range(0, len(mints), 30):
        chunk = mints[i:i + 30]
        data = _get(f"{DEXSCREENER}/tokens/v1/solana/{','.join(chunk)}")
        if not data:
            continue
        pairs = data if isinstance(data, list) else data.get("pairs") or []
        for p in pairs:
            mint = (p.get("baseToken") or {}).get("address")
            price = _f(p.get("priceUsd"))
            liq = _f((p.get("liquidity") or {}).get("usd"))
            if mint and price:
                # Keep the most liquid pair's price per mint
                if mint not in prices or liq > prices[mint][1]:
                    prices[mint] = (price, liq)
    return {m: p for m, (p, _) in prices.items()}


def rugcheck_report(mint):
    """On-chain safety report. Returns a normalized dict or None on failure
    (callers must treat None as fail-closed for buys)."""
    data = _get(f"{RUGCHECK}/tokens/{mint}/report", timeout=25)
    if not data or not isinstance(data, dict):
        return None

    token = data.get("token") or {}
    holders = data.get("topHolders") or []
    # Exclude AMM/locked accounts flagged as insiders=False LP where possible;
    # rugcheck marks LP accounts with 'owner' fields we can't fully trust, so
    # we take the raw top-10 share as a conservative upper bound.
    top10 = sum(_f(h.get("pct")) for h in holders[:10])

    risks = [r.get("name", "") for r in (data.get("risks") or [])]
    danger = [r.get("name", "") for r in (data.get("risks") or [])
              if str(r.get("level", "")).lower() == "danger"]

    markets = data.get("markets") or []
    lp_locked_pct = 0.0
    for m in markets:
        lp = m.get("lp") or {}
        lp_locked_pct = max(lp_locked_pct, _f(lp.get("lpLockedPct")))

    return {
        "mint_authority_active": bool(token.get("mintAuthority")),
        "freeze_authority_active": bool(token.get("freezeAuthority")),
        "top10_pct": top10,
        "lp_locked_pct": lp_locked_pct,
        "total_holders": int(data.get("totalHolders") or 0),
        "risks": risks,
        "danger_risks": danger,
        "rugcheck_score": _f(data.get("score_normalised", data.get("score"))),
    }
