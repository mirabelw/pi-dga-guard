"""Time the complete portable scorer on this machine; run again on your Pi."""

import argparse
import csv
import json
from pathlib import Path
import platform
import resource
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dgaguard.model import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="domain,label CSV to replay")
    parser.add_argument("--model", default="models/dga_model.json")
    parser.add_argument("--limit", type=int, default=10000)
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("limit must be positive")
    with Path(args.csv).open(newline="", encoding="utf-8") as stream:
        domains = [row["domain"] for _, row in zip(range(args.limit), csv.DictReader(stream))]
    start = time.perf_counter()
    model = load_model(args.model)
    load_seconds = time.perf_counter() - start
    start = time.perf_counter()
    scores = [model.score(domain) for domain in domains]
    score_seconds = time.perf_counter() - start
    valid = sum(score is not None for score in scores)
    if not valid:
        raise ValueError("no scoreable domains in CSV")
    print(json.dumps({
        "architecture": platform.machine(),
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "domains": len(domains), "scored": valid,
        "model_load_seconds": round(load_seconds, 3),
        "scoring_microseconds_per_domain": round(score_seconds * 1e6 / len(domains), 2),
        "domains_per_second": round(len(domains) / score_seconds),
        "model_json_bytes": Path(args.model).stat().st_size,
        "peak_process_rss_kib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "note": ("Single process run, including Python baseline memory and suffix-list load; "
                 "excludes SQLite polling, log writes, and concurrent DNS traffic. "
                 "Repeat on the target Raspberry Pi."),
    }, indent=2))


if __name__ == "__main__":
    main()
