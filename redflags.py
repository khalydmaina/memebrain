"""Reflex layer: deterministic red-flag gates.

These run before any scoring or AI and are absolute — a candidate failing
any gate is rejected no matter how good it looks otherwise. Money rules
live in code, never in a model. If safety data cannot be fetched, the
candidate FAILS CLOSED: unknown safety is treated as unsafe.
"""

import config

# rugcheck risk names that are instant disqualifiers regardless of level
FATAL_RISKS = {
    "Freeze Authority still enabled",
    "Mint Authority still enabled",
    "Single holder ownership",
    "High ownership",
    "Copycat token",
    "Low amount of LP Providers",
}


def check(candidate, report):
    """Returns (passed: bool, flags: list[str]). candidate is the pool dict
    from sources.discover_pools, report from sources.rugcheck_report (may
    be None)."""
    flags = []

    liq = candidate.get("liquidity_usd") or 0
    if liq < config.MIN_LIQUIDITY_USD:
        flags.append(f"liquidity ${liq:,.0f} < ${config.MIN_LIQUIDITY_USD:,.0f}")

    if (candidate.get("vol_h1") or 0) < config.MIN_VOLUME_H1_USD:
        flags.append(f"1h volume ${candidate.get('vol_h1', 0):,.0f} too thin")

    age = candidate.get("age_hours")
    if age is None:
        flags.append("unknown pool age")
    elif age > config.MAX_TOKEN_AGE_HOURS:
        flags.append(f"too old ({age:.0f}h)")

    sells = candidate.get("sells_h1") or 0
    buys = candidate.get("buys_h1") or 0
    if buys + sells >= 20 and sells == 0:
        # nobody can sell = honeypot signature
        flags.append("zero sells despite activity (honeypot pattern)")

    if report is None:
        flags.append("safety report unavailable (fail closed)")
        return False, flags

    if report["mint_authority_active"]:
        flags.append("mint authority NOT revoked (infinite supply risk)")
    if report["freeze_authority_active"]:
        flags.append("freeze authority active (your tokens can be frozen)")
    if report["top10_pct"] > config.MAX_TOP10_HOLDER_PCT:
        flags.append(f"top10 holders own {report['top10_pct']:.0f}%")
    if report["total_holders"] and report["total_holders"] < 50:
        flags.append(f"only {report['total_holders']} holders")

    fatal = FATAL_RISKS.intersection(report["danger_risks"])
    for name in fatal:
        flags.append(f"rugcheck danger: {name}")
    # any danger-level risk not on our fatal list still fails the gate;
    # unknown dangers are dangers
    for name in report["danger_risks"]:
        if name not in fatal:
            flags.append(f"rugcheck danger: {name}")

    return len(flags) == 0, flags
