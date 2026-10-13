"""Validation-only model selection for the character model; test splits are never read.

Each variant trains on the experiment's training split and is scored on its validation split.
Primary criterion: recall on the unseen-family validation split at a threshold allowing 0.1% benign
false positives (average precision breaks ties). The threshold is set on the same split, so absolute
values are optimistic but the ranking is fair.

    python scripts/experiments.py --data-dir data --wordlist data/english_ranked.txt
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dgaguard.features import build_lexicon, extract, lexical_label, normalize_domain  # noqa: E402
from dgaguard.suffix import public_suffix  # noqa: E402

TARGET_FPR = 0.001


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            domain = normalize_domain(row["domain"])
            token = lexical_label(domain) if domain else None
            if token is None:
                continue
            rows.append({"domain": domain, "token": token, "suffix": public_suffix(domain),
                         "label": int(row["label"] in {"1", "dga"}), "family": row.get("family", "")})
    return rows


def recall_at_fpr(labels, scores, target=TARGET_FPR):
    benign = sorted((s for y, s in zip(labels, scores) if y == 0), reverse=True)
    allowed = max(1, int(math.floor(target * len(benign))))
    threshold = benign[allowed - 1] + 1e-12
    positives = [s for y, s in zip(labels, scores) if y == 1]
    return sum(s >= threshold for s in positives) / len(positives)


def build(variant, lexicon):
    import numpy as np
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    wrap = (lambda t: f"^{t}$") if variant.get("boundary") else (lambda t: t)
    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=variant.get("ngrams", (2, 5)), lowercase=False,
        min_df=3, max_features=variant.get("max_features", 40000),
        sublinear_tf=variant.get("sublinear", False), dtype=np.float32)
    scaler, encoder = StandardScaler(), OneHotEncoder(handle_unknown="ignore", min_frequency=20)

    def transform(rows, fit=False):
        parts = []
        if variant.get("char", True):
            texts = [wrap(r["token"]) for r in rows]
            parts.append(vectorizer.fit_transform(texts) if fit else vectorizer.transform(texts))
        if variant.get("handcrafted"):
            dense = [extract(r["domain"], lexicon) for r in rows]
            dense = scaler.fit_transform(dense) if fit else scaler.transform(dense)
            parts.append(csr_matrix(dense * variant.get("handcrafted_weight", 1.0)))
        if variant.get("suffix"):
            values = [[r["suffix"]] for r in rows]
            parts.append(encoder.fit_transform(values) if fit else encoder.transform(values))
        return hstack(parts).tocsr() if len(parts) > 1 else parts[0]
    return transform


def run(variant, splits, lexicon):
    from sklearn.linear_model import LogisticRegression
    results = {}
    for name, (train, validation) in splits.items():
        transform = build(variant, lexicon)
        x_train = transform(train, fit=True)
        clf = LogisticRegression(C=variant.get("C", 20.0), class_weight="balanced",
                                 max_iter=2000, random_state=42)
        clf.fit(x_train, [r["label"] for r in train])
        scores = clf.predict_proba(transform(validation))[:, 1]
        labels = [r["label"] for r in validation]
        from sklearn.metrics import average_precision_score
        results[name] = {"recall_at_0.1pct_fpr": round(recall_at_fpr(labels, scores), 4),
                         "average_precision": round(float(average_precision_score(labels, scores)), 4)}
        if name == "family":
            per_family = {}
            benign = sorted((s for y, s in zip(labels, scores) if y == 0), reverse=True)
            threshold = benign[max(1, int(TARGET_FPR * len(benign))) - 1] + 1e-12
            for r, s in zip(validation, scores):
                if r["label"]:
                    hit, total = per_family.get(r["family"], (0, 0))
                    per_family[r["family"]] = (hit + (s >= threshold), total + 1)
            results[name]["per_family"] = {k: f"{h}/{t}" for k, (h, t) in sorted(per_family.items())}
    return results


VARIANTS = {
    "linear_7_handcrafted_only": {"char": False, "handcrafted": True, "C": 1.0},
    "char_2_5_C20 (previous default)": {},
    "+boundary": {"boundary": True},
    "+boundary+sublinear": {"boundary": True, "sublinear": True},
    "+boundary C=5": {"boundary": True, "C": 5.0},
    "+boundary C=100": {"boundary": True, "C": 100.0},
    "+boundary ngrams 1-5": {"boundary": True, "ngrams": (1, 5)},
    "+boundary ngrams 2-6": {"boundary": True, "ngrams": (2, 6)},
    "+boundary ngrams 3-6": {"boundary": True, "ngrams": (3, 6)},
    "+boundary 150k features": {"boundary": True, "max_features": 150000},
    "+boundary+handcrafted": {"boundary": True, "handcrafted": True},
    "+boundary+handcrafted(x0.3)": {"boundary": True, "handcrafted": True, "handcrafted_weight": 0.3},
    "+boundary+suffix": {"boundary": True, "suffix": True},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--wordlist", default="data/english_ranked.txt")
    parser.add_argument("--only", nargs="*", help="run only these variant names")
    parser.add_argument("--extra", help="JSON object of extra {name: variant} to run")
    parser.add_argument("--out", default="results/model_selection.json")
    parser.add_argument("--modes", nargs="+", default=["family", "known"], choices=["family", "known"])
    args = parser.parse_args()
    data = Path(args.data_dir)
    splits = {mode: (load(data / f"{mode}_train.csv"), load(data / f"{mode}_validation.csv"))
              for mode in args.modes}
    lexicon = build_lexicon(Path(args.wordlist).read_text(encoding="utf-8").splitlines())
    lexicon["words"] = set(lexicon["words"])
    variants = dict(VARIANTS)
    if args.extra:
        extra = json.loads(args.extra)
        args.only = (args.only or []) + list(extra)
        variants.update({k: {kk: tuple(vv) if isinstance(vv, list) else vv for kk, vv in v.items()}
                         for k, v in extra.items()})
    if args.only:
        variants = {k: v for k, v in variants.items() if k in args.only}
    out = Path(args.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    for name, variant in variants.items():
        start = time.time()
        entry = results.setdefault(name, {})
        entry["config"] = {k: list(v) if isinstance(v, tuple) else v for k, v in variant.items()}
        entry.update(run(variant, splits, lexicon))
        summary = " | ".join(f"{mode}-val R@0.1%={entry[mode]['recall_at_0.1pct_fpr']:.3f} "
                             f"AP={entry[mode]['average_precision']:.3f}" for mode in args.modes)
        print(f"{name:34} {summary} {entry.get('family', {}).get('per_family', '')} "
              f"({time.time() - start:.0f}s)", flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
