"""Offline training for the 7-feature int8 linear baseline, kept as an ablation against the n-gram model."""

import json
from pathlib import Path

from .features import FEATURE_NAMES, build_lexicon, extract
from .model import Int8Model
from .suffix import SUFFIX_RULES
from .train_char import evaluate, read_labeled, threshold_at_fpr


def train(csv_path, model_path, report_path, wordlist, validation_csv=None, test_csv=None,
          seed=42, max_words=20000, target_fpr=0.001):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    rows = read_labeled(csv_path)
    if validation_csv:
        training, validation = rows, read_labeled(validation_csv)
        testing = read_labeled(test_csv) if test_csv else None
    else:
        training, remainder = train_test_split(
            rows, test_size=.30, random_state=seed, stratify=[r["label"] for r in rows])
        validation, testing = train_test_split(
            remainder, test_size=.50, random_state=seed, stratify=[r["label"] for r in remainder])
    token_sets = [{r["token"] for r in split} for split in (training, validation, testing or [])]
    if (token_sets[0] & token_sets[1]) or (token_sets[0] & token_sets[2]) or (token_sets[1] & token_sets[2]):
        raise ValueError("lexical token overlap across splits; group by registrable label")

    lexicon = build_lexicon(Path(wordlist).read_text(encoding="utf-8").splitlines(), limit=max_words)
    lexicon["words"] = set(lexicon["words"])
    x_train = [extract(r["domain"], lexicon) for r in training]
    scaler = StandardScaler().fit(x_train)
    classifier = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    classifier.fit(scaler.transform(x_train), [r["label"] for r in training])
    # Symmetric per-tensor int8. The input range comes from training data only.
    input_range = max(8.0, float(abs(scaler.transform(x_train)).max()))
    weights = classifier.coef_[0]
    weight_scale = max(abs(float(w)) for w in weights) / 127.0
    if weight_scale == 0:
        raise ValueError("classifier learned all-zero coefficients")
    artifact = {
        "format": "dgaguard-linear-int8-v2",
        "suffix_rules": SUFFIX_RULES,
        "features": list(FEATURE_NAMES),
        "mean": [float(v) for v in scaler.mean_],
        "std": [float(v) for v in scaler.scale_],
        "input_scale": input_range / 127.0,
        "weight_scale": weight_scale,
        "weights_int8": [max(-127, min(127, round(float(w) / weight_scale))) for w in weights],
        "bias": float(classifier.intercept_[0]),
        "alert_threshold": 0.5,  # replaced from validation below
        "lexicon": {"words": sorted(lexicon["words"]), "bigram_logp": lexicon["bigram_logp"]},
    }
    model = Int8Model(artifact)
    validation_scores = [model.score(r["domain"]) for r in validation]
    threshold = threshold_at_fpr([r["label"] for r in validation], validation_scores, target_fpr)
    artifact["alert_threshold"] = threshold
    float_validation = classifier.predict_proba(
        scaler.transform([extract(r["domain"], lexicon) for r in validation]))[:, 1]
    report = {
        "model": "handcrafted_7_feature_logistic_regression_int8",
        "suffix_rules": SUFFIX_RULES,
        "parameters": {"features": list(FEATURE_NAMES), "wordlist_size": len(lexicon["words"]),
                       "target_validation_fpr": target_fpr, "seed": seed},
        "training_count": len(training),
        "int8_vs_float_max_abs_difference": float(max(abs(a - b) for a, b in zip(float_validation, validation_scores))),
        "float_weights_standardized": [float(w) for w in weights],
        "validation": evaluate(validation, validation_scores, threshold),
        "limitations": "Baseline for comparison with the character model; scores are not calibrated probabilities.",
    }
    if testing:
        report["test"] = evaluate(testing, [model.score(r["domain"]) for r in testing], threshold)
    for path, content in ((model_path, json.dumps(artifact, separators=(",", ":"))),
                          (report_path, json.dumps(report, indent=2) + "\n")):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content, encoding="utf-8")
    return report
