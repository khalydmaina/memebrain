"""Outcome tracker: turns decisions into training labels.

Every decision (buy or skip) gets its price sampled at checkpoints after
the decision. When the last checkpoint lands, the row is labeled:
  win  — price reached WIN_THRESHOLD at any checkpoint and never rugged
  loss — rugged, or fell to LOSS_THRESHOLD before ever winning
  flat — everything else
A token whose price disappears from every feed is treated as rugged:
in this market, vanishing IS an outcome, not missing data.
"""

from datetime import datetime, timezone

import config
import db
import sources

CHECKPOINT_FIELDS = {5: "m5", 30: "m30", 120: "m120", 1440: "m1440"}
_MISS_LIMIT = 3
_miss_counts = {}


def _label(row_dict):
    multiples = [row_dict[f] for f in CHECKPOINT_FIELDS.values()
                 if row_dict.get(f) is not None]
    if row_dict.get("rugged"):
        return "loss"
    if not multiples:
        return None
    if max(multiples) >= config.WIN_THRESHOLD:
        return "win"
    if min(multiples) <= config.LOSS_THRESHOLD:
        return "loss"
    return "flat"


def process(con, log):
    """Sample due checkpoints for all pending outcomes. One batched price
    call covers every mint."""
    pending = db.pending_outcomes(con)
    if not pending:
        return 0

    now = datetime.now(timezone.utc)
    due = []
    for row in pending:
        decided = datetime.fromisoformat(row["decision_ts"])
        age_min = (now - decided).total_seconds() / 60
        for minutes, field in CHECKPOINT_FIELDS.items():
            if row[field] is None and age_min >= minutes:
                due.append((row, field, minutes))

    if not due:
        return 0

    mints = list({row["mint"] for row, _, _ in due})
    prices = sources.batch_prices(mints)

    updated = 0
    for row, field, minutes in due:
        price = prices.get(row["mint"])
        did = row["decision_id"]

        if price is None:
            # Token gone from the feed. Tolerate transient feed gaps, then
            # call it: dead token = rug.
            _miss_counts[did] = _miss_counts.get(did, 0) + 1
            if _miss_counts[did] >= _MISS_LIMIT:
                db.update_outcome(con, did, **{field: 0.0}, rugged=1)
            else:
                continue
        else:
            multiple = price / row["base_price"] if row["base_price"] else 0.0
            fields = {field: round(multiple, 4)}
            if multiple <= config.RUG_THRESHOLD:
                fields["rugged"] = 1
            db.update_outcome(con, did, **fields)

        # Re-read and finalize if the last checkpoint is in
        fresh = con.execute("SELECT * FROM outcomes WHERE decision_id=?",
                            (did,)).fetchone()
        fresh = dict(fresh)
        if fresh["m1440"] is not None or fresh["rugged"]:
            db.update_outcome(con, did, label=_label(fresh), done=1)
            _miss_counts.pop(did, None)
        updated += 1

    return updated
