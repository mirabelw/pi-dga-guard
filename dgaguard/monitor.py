"""Read-only polling of the Pi-hole long-term query DB."""

import json
import logging
import os
from contextlib import closing
from pathlib import Path
import sqlite3
import time

from .features import DEFAULT_LOCAL_SUFFIXES, normalize_domain
from .policy import is_allowlisted

LOG = logging.getLogger(__name__)
# Scores are intended for queries Pi-hole allowed or forwarded, not known blocks.
ALLOWED_STATUSES = (2, 3, 12, 13, 14, 17)


def open_query_db(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
    connection.execute("PRAGMA query_only=ON")
    return connection


def read_batch(connection, after_id, limit):
    return connection.execute(
        "SELECT id, timestamp, domain, status FROM queries "
        "WHERE id > ? ORDER BY id LIMIT ?", (after_id, limit)
    ).fetchall()


def max_query_id(connection):
    return connection.execute("SELECT COALESCE(MAX(id),0) FROM queries").fetchone()[0]


def write_json_atomic(path, data):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def log_event(path, event):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, separators=(",", ":")) + "\n")


def poll_once(model, db_path, state_path, alerts_path, *, alert_threshold=0.90,
              block_threshold=0.995, autoblock=None, backfill=False,
              batch_size=500, cooldown_seconds=86400, allowlist=frozenset(),
              local_suffixes=DEFAULT_LOCAL_SUFFIXES):
    db_path = Path(db_path)
    stat = db_path.stat()
    identity = f"{stat.st_dev}:{stat.st_ino}"
    try:
        state = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        state = {}
    with closing(open_query_db(db_path)) as connection:
        if state.get("db_identity") != identity:
            start = 0 if backfill else max_query_id(connection)
            state = {"db_identity": identity, "last_id": start, "alerts": {}}
            write_json_atomic(state_path, state)
            if not backfill:
                LOG.info("Starting from current database end (id=%s)", start)
        rows = read_batch(connection, state["last_id"], batch_size)
    now = time.time()
    state["alerts"] = {domain: stamp for domain, stamp in state["alerts"].items()
                       if now - stamp < cooldown_seconds}
    for row_id, timestamp, raw_domain, status in rows:
        state["last_id"] = row_id
        domain = normalize_domain(raw_domain, local_suffixes=local_suffixes)
        if status not in ALLOWED_STATUSES or domain is None:
            continue
        if is_allowlisted(domain, allowlist):
            continue
        if domain in state["alerts"]:
            continue
        score = model.score(domain)
        if score is None or score < alert_threshold:
            continue
        event = {"timestamp": timestamp, "query_id": row_id,
                 "domain": domain, "score": round(score, 6),
                 "action": "alert"}
        if autoblock and score >= block_threshold:
            try:
                autoblock.deny_exact(domain)
                event["action"] = "blocked"
            except Exception as exc:
                event["action"] = "block_failed"
                LOG.error("Failed to deny %s: %s", domain, exc)
        log_event(alerts_path, event)
        LOG.warning("%s", event)
        # Retry an API failure on a subsequent query; suppress successful alerts.
        if event["action"] != "block_failed":
            state["alerts"][domain] = now
    write_json_atomic(state_path, state)
    return len(rows)
