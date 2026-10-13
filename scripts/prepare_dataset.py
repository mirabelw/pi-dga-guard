"""Reproducible, registrable-label-disjoint train/validation/test splits.

Input is chrmor/DGA_domains_dataset (no header; class,family,domain), which is not redistributed here.
"""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dgaguard.features import lexical_label, normalize_domain  # noqa: E402


def fraction(key, seed, namespace=""):
    raw = f"{namespace}{seed}:{key}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") / 2**64


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source, output_dir, mode, seed, full_train=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_families = {"corebot", "gozi", "suppobox"}
    test_families = {"matsnu", "necurs", "pykspa"}
    rows = {"train": [], "validation": [], "test": []}
    seen_domains = set()
    with Path(source).open(newline="", encoding="utf-8") as stream:
        for kind, family, raw_domain in csv.reader(stream):
            domain = normalize_domain(raw_domain)
            if domain is None or domain in seen_domains:
                continue
            token = lexical_label(domain)
            if token is None:
                continue
            seen_domains.add(domain)
            r = (fraction(token, seed, "random:") if mode == "known"
                 else fraction(domain, seed))
            # Validation/test ranges never change; --full-train only adds the
            # otherwise-unused remainder of the data to training.
            if mode == "known":
                split = ("train" if r < .17 else "validation" if r < .21
                         else "test" if r < .25 else "train" if full_train else None)
            elif kind == "legit":
                split = ("train" if r < .15 else "validation" if r < .19
                         else "test" if r < .23 else "train" if full_train else None)
            elif family in validation_families:
                split = "validation" if r < .30 else None
            elif family in test_families:
                split = "test" if r < .30 else None
            else:
                split = "train" if r < .18 or full_train else None
            if split is not None:
                if kind not in {"legit", "dga"}:
                    raise ValueError(f"unknown source class {kind!r}")
                rows[split].append((domain, "dga" if kind == "dga" else "benign",
                                    family, token))

    # Family assignment can collide at a lexical label with a different
    # family or Alexa domain. Retain training, then validation, then test.
    train_tokens = {row[3] for row in rows["train"]}
    before_val = len(rows["validation"])
    rows["validation"] = [row for row in rows["validation"] if row[3] not in train_tokens]
    used_tokens = train_tokens | {row[3] for row in rows["validation"]}
    before_test = len(rows["test"])
    rows["test"] = [row for row in rows["test"] if row[3] not in used_tokens]

    counts = {}
    for split, records in rows.items():
        path = output_dir / f"{mode}_{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["domain", "label", "family"])
            writer.writerows(row[:3] for row in records)
        counts[split] = {
            "total": len(records),
            "benign": sum(row[1] == "benign" for row in records),
            "dga": sum(row[1] == "dga" for row in records),
            "dga_families": dict(sorted(Counter(row[2] for row in records
                                                if row[1] == "dga").items())),
        }
    psl = Path(__file__).resolve().parents[1] / "data" / "public_suffix_list.dat"
    manifest = {
        "source_url": "https://github.com/chrmor/DGA_domains_dataset/blob/master/dga_domains_full.csv",
        "source_sha256": file_sha256(source),
        "psl_sha256": file_sha256(psl),
        "mode": mode, "seed": seed, "full_train": full_train,
        "validation_families": sorted(validation_families) if mode == "family" else [],
        "test_families": sorted(test_families) if mode == "family" else [],
        "removed_validation_token_collisions": before_val - len(rows["validation"]),
        "removed_test_token_collisions": before_test - len(rows["test"]),
        "counts": counts,
    }
    (output_dir / f"{mode}_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--mode", choices=("known", "family"), required=True)
    parser.add_argument("--seed", type=int, help="default 83 for known, 73 for family")
    parser.add_argument("--full-train", action="store_true",
                        help="also train on data outside the fixed validation/test ranges")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (83 if args.mode == "known" else 73)
    result = prepare(args.source, args.output_dir, args.mode, seed, full_train=args.full_train)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
