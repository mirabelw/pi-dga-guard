"""Offline character TF-IDF training; the Pi needs no scikit-learn."""

import csv
import json
import math
from pathlib import Path

from .char_model import FORMAT, CharModel, wrap
from .features import lexical_label, normalize_domain
from .suffix import SUFFIX_RULES


def read_labeled(path):
    records = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            domain = normalize_domain(row["domain"])
            token = lexical_label(domain) if domain else None
            if token is None:
                continue
            value = row["label"].strip().lower()
            if value not in {"0", "1", "benign", "dga"}:
                raise ValueError(f"unknown label {value!r}")
            label = int(value in {"1", "dga"})
            if domain in records and records[domain]["label"] != label:
                raise ValueError(f"conflicting labels for {domain}")
            records[domain] = {"domain": domain, "token": token,
                               "label": label, "family": row.get("family", "")}
    if len(records) < 20 or len({r["label"] for r in records.values()}) != 2:
        raise ValueError("need at least 20 unique domains and both classes")
    return list(records.values())


def threshold_at_fpr(labels, scores, target):
    if not 0 < target < 1:
        raise ValueError("target FPR must be in (0, 1)")
    benign = sorted((float(score) for label, score in zip(labels, scores) if label == 0),
                    reverse=True)
    if not benign:
        raise ValueError("validation split needs benign domains")
    allowed = max(1, int(math.floor(target * len(benign))))
    return min(1.0, benign[allowed - 1] + 1e-12)


def evaluate(records, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score
    y = [r["label"] for r in records]
    tp = sum(label == 1 and score >= threshold for label, score in zip(y, scores))
    fp = sum(label == 0 and score >= threshold for label, score in zip(y, scores))
    positives, negatives = sum(y), len(y) - sum(y)
    by_family = {}
    for record, score in zip(records, scores):
        if record["label"] != 1:
            continue
        family = record["family"] or "unlabeled"
        item = by_family.setdefault(family, {"positives": 0, "detected": 0})
        item["positives"] += 1
        item["detected"] += score >= threshold
    false_positives = [
        {"domain": r["domain"], "score": round(float(s), 6)}
        for r, s in zip(records, scores) if r["label"] == 0 and s >= threshold
    ]
    false_positives.sort(key=lambda entry: entry["score"], reverse=True)
    return {
        "count": len(y), "positives": positives, "benign": negatives,
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "threshold": threshold, "tp": tp, "fp": fp,
        "recall": tp / positives, "fpr": fp / negatives,
        "precision_on_this_dataset": tp / (tp + fp) if tp + fp else None,
        "families": dict(sorted(by_family.items())),
        "false_positive_domains": false_positives[:1000],
        "false_positives_truncated": len(false_positives) > 1000,
    }


def train(csv_path, model_path, report_path, validation_csv=None, test_csv=None,
          seed=42, c=20.0, max_features=150000, ngram_min=1, ngram_max=6,
          target_fpr=0.001):
    """Defaults come from validation-only selection; see scripts/experiments.py."""
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    if not 1 <= ngram_min <= ngram_max <= 8 or max_features <= 0 or c <= 0:
        raise ValueError("invalid model hyperparameters")
    rows = read_labeled(csv_path)
    if validation_csv:
        validation = read_labeled(validation_csv)
        testing = read_labeled(test_csv) if test_csv else None
        training = rows
    else:
        # Demo convenience only. For real evaluation, supply explicit family/time-aware
        # validation and test CSVs.
        training, remainder = train_test_split(
            rows, test_size=.30, random_state=seed,
            stratify=[r["label"] for r in rows])
        validation, testing = train_test_split(
            remainder, test_size=.50, random_state=seed,
            stratify=[r["label"] for r in remainder])
    splits = [training, validation] + ([testing] if testing else [])
    token_sets = [{r["token"] for r in split} for split in splits]
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            if token_sets[i] & token_sets[j]:
                raise ValueError("lexical token overlap across splits; group by registrable label")

    vectorizer = TfidfVectorizer(
        analyzer="char", ngram_range=(ngram_min, ngram_max),
        min_df=3, max_features=max_features, lowercase=False, dtype=np.float32,
    )
    x_train = vectorizer.fit_transform([wrap(r["token"]) for r in training])
    clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000,
                             random_state=seed)
    clf.fit(x_train, [r["label"] for r in training])
    # 7 significant digits keeps the model JSON compact.
    weights = {
        term: [float(f"{clf.coef_[0, index] * vectorizer.idf_[index]:.7g}"),
               float(f"{vectorizer.idf_[index] ** 2:.7g}")]
        for term, index in vectorizer.vocabulary_.items()
    }
    artifact = {
        "format": FORMAT,
        "suffix_rules": SUFFIX_RULES,
        "ngram_range": [ngram_min, ngram_max],
        "bias": float(clf.intercept_[0]),
        "ngrams": weights,
        "alert_threshold": 0.5,  # filled from validation only
    }
    model = CharModel(artifact)
    validation_scores = clf.predict_proba(
        vectorizer.transform([wrap(r["token"]) for r in validation]))[:, 1]
    threshold = threshold_at_fpr([r["label"] for r in validation],
                                 validation_scores, target_fpr)
    artifact["alert_threshold"] = threshold
    val_runtime = [model.score(r["domain"]) for r in validation]
    report = {
        "model": "character_tfidf_logistic_regression",
        "suffix_rules": SUFFIX_RULES,
        "parameters": {"C": c, "ngram_range": [ngram_min, ngram_max],
                       "max_features": max_features, "vocabulary_size": len(weights),
                       "target_validation_fpr": target_fpr, "seed": seed},
        "training_count": len(training),
        "portable_score_max_abs_difference": float(max(abs(a - b) for a, b in zip(validation_scores, val_runtime))),
        "validation": evaluate(validation, val_runtime, threshold),
        "limitations": ("The score is not a calibrated probability. Domain-level splitting of known "
                        "families does not establish unseen-family or live-network performance."),
    }
    if testing:
        test_scores = [model.score(r["domain"]) for r in testing]
        report["test"] = evaluate(testing, test_scores, threshold)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(model_path).write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    Path(report_path).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report
