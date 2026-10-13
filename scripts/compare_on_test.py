"""One-time test evaluation of a scripts/experiments.py variant, at a threshold set on validation
(0.1% benign false positives). Run only after model selection; see --help for arguments."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.experiments import VARIANTS, build, load  # noqa: E402
from dgaguard.features import build_lexicon  # noqa: E402
from dgaguard.train_char import threshold_at_fpr  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--test", required=True)
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--wordlist", default="data/english_ranked.txt")
    parser.add_argument("--target-fpr", type=float, default=0.001)
    args = parser.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    variant = VARIANTS[args.variant]
    lexicon = None
    if variant.get("handcrafted"):
        lexicon = build_lexicon(Path(args.wordlist).read_text(encoding="utf-8").splitlines())
        lexicon["words"] = set(lexicon["words"])
    train, validation, test = load(args.train), load(args.validation), load(args.test)
    transform = build(variant, lexicon)
    clf = LogisticRegression(C=variant.get("C", 20.0), class_weight="balanced",
                             max_iter=2000, random_state=42)
    clf.fit(transform(train, fit=True), [r["label"] for r in train])
    val_scores = clf.predict_proba(transform(validation))[:, 1]
    threshold = threshold_at_fpr([r["label"] for r in validation], val_scores, args.target_fpr)
    scores = clf.predict_proba(transform(test))[:, 1]
    labels = [r["label"] for r in test]
    families = {}
    for r, s in zip(test, scores):
        if r["label"]:
            hit, total = families.get(r["family"], (0, 0))
            families[r["family"]] = (hit + int(s >= threshold), total + 1)
    print(json.dumps({
        "variant": args.variant, "training_count": len(train), "threshold": threshold,
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "tp": int(sum(y == 1 and s >= threshold for y, s in zip(labels, scores))),
        "positives": sum(labels),
        "fp": int(sum(y == 0 and s >= threshold for y, s in zip(labels, scores))),
        "benign": len(labels) - sum(labels),
        "families": {k: f"{h}/{t}" for k, (h, t) in sorted(families.items())},
    }, indent=2))


if __name__ == "__main__":
    main()
