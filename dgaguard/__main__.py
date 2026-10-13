import argparse
import json
import logging
import sqlite3
import time

from .model import load_model


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pi-hole DGA detector")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train off-device and export a portable character model")
    train.add_argument("--csv", required=True, help="CSV with domain,label columns")
    train.add_argument("--model", default="artifacts/model.json")
    train.add_argument("--report", default="artifacts/report.json")
    train.add_argument("--validation-csv", help="disjoint data used to set an alert threshold")
    train.add_argument("--test-csv", help="final disjoint evaluation data")
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--model-type", choices=("char", "linear"), default="char",
                       help="char: TF-IDF n-gram model (default); linear: 7-feature int8 baseline")
    train.add_argument("--wordlist", default="data/english_ranked.txt",
                       help="frequency-ranked English words (linear baseline only)")
    train.add_argument("--c", type=float, default=20.0, help="inverse regularization strength (char model)")
    train.add_argument("--max-features", type=int, default=150000)
    train.add_argument("--ngram-min", type=int, default=1)
    train.add_argument("--ngram-max", type=int, default=6)
    train.add_argument("--target-fpr", type=float, default=0.001,
                       help="validation benign false-positive rate for default alert threshold")

    score = commands.add_parser("score", help="inspect one domain")
    score.add_argument("domain")
    score.add_argument("--model", default="models/dga_model.json")

    monitor = commands.add_parser("monitor", help="poll Pi-hole long-term SQLite log")
    monitor.add_argument("--model", default="models/dga_model.json")
    monitor.add_argument("--db", default="/etc/pihole/pihole-FTL.db")
    monitor.add_argument("--state", default="state/cursor.json")
    monitor.add_argument("--alerts", default="state/alerts.jsonl")
    monitor.add_argument("--interval", type=float, default=10.0)
    monitor.add_argument("--batch-size", type=int, default=500)
    monitor.add_argument("--alert-threshold", type=float,
                         help="override threshold selected on validation benign domains")
    monitor.add_argument("--block-threshold", type=float,
                         help="required if --auto-block is enabled")
    monitor.add_argument("--cooldown", type=int, default=86400)
    monitor.add_argument("--backfill", action="store_true", help="on first run, process the historical DB")
    monitor.add_argument("--once", action="store_true", help="process one batch and exit")
    monitor.add_argument("--auto-block", action="store_true", help="enable Pi-hole v6 exact denylist writes")
    monitor.add_argument("--api-url", default="http://127.0.0.1")
    monitor.add_argument("--allowlist", help="plain domain list or Tranco rank,domain CSV")
    monitor.add_argument("--local-suffix", action="append", default=[],
                         help="additional private DNS suffix to skip (repeatable)")

    args = parser.parse_args(argv)
    if args.command == "train" and args.model_type == "linear":
        from .train import train as train_linear
        print(json.dumps(train_linear(args.csv, args.model, args.report, args.wordlist,
                                      validation_csv=args.validation_csv, test_csv=args.test_csv,
                                      seed=args.seed, target_fpr=args.target_fpr), indent=2))
        return
    if args.command == "train":
        from .train_char import train as train_model
        print(json.dumps(train_model(args.csv, args.model, args.report,
                                     validation_csv=args.validation_csv,
                                     test_csv=args.test_csv, seed=args.seed,
                                     c=args.c, max_features=args.max_features,
                                     ngram_min=args.ngram_min, ngram_max=args.ngram_max,
                                     target_fpr=args.target_fpr), indent=2))
        return
    model = load_model(args.model)
    if args.command == "score":
        value = model.score(args.domain)
        if value is None:
            parser.error("invalid, local, or unsupported domain")
        print(f"{args.domain}: DGA score {value:.6f} (not a calibrated probability)")
        return

    threshold = args.alert_threshold if args.alert_threshold is not None else model.default_alert_threshold
    if args.auto_block and args.block_threshold is None:
        parser.error("--auto-block requires an explicit --block-threshold")
    block_threshold = args.block_threshold if args.block_threshold is not None else 1.0
    if not 0 <= threshold <= block_threshold <= 1:
        parser.error("require 0 <= alert-threshold <= block-threshold <= 1")
    if args.interval <= 0 or args.batch_size <= 0 or args.cooldown < 0:
        parser.error("interval, batch size, and cooldown must be valid positive values")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    from .monitor import poll_once
    from .features import DEFAULT_LOCAL_SUFFIXES, VALID_LABEL
    from .policy import load_allowlist
    local_suffixes = tuple(DEFAULT_LOCAL_SUFFIXES) + tuple(
        suffix.strip().lower().lstrip(".") for suffix in args.local_suffix
    )
    if any(not suffix or any(not VALID_LABEL.fullmatch(label) for label in suffix.split("."))
           for suffix in local_suffixes):
        parser.error("invalid local suffix")
    allowlist = load_allowlist(args.allowlist) if args.allowlist else frozenset()
    autoblock = None
    if args.auto_block:
        from .pihole_api import PiholeAPI
        autoblock = PiholeAPI(args.api_url)
        autoblock.authenticate()
    while True:
        try:
            count = poll_once(model, args.db, args.state, args.alerts,
                              alert_threshold=threshold,
                              block_threshold=block_threshold,
                              autoblock=autoblock, backfill=args.backfill,
                              batch_size=args.batch_size, cooldown_seconds=args.cooldown,
                              allowlist=allowlist, local_suffixes=local_suffixes)
        except (OSError, sqlite3.Error) as exc:
            logging.error("Pi-hole database unavailable: %s", exc)
            if args.once:
                raise SystemExit(1) from exc
            count = 0
        if args.once:
            print(f"Processed {count} rows")
            return
        time.sleep(0.1 if count == args.batch_size else args.interval)


if __name__ == "__main__":
    main()
