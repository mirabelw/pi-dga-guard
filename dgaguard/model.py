"""Standard-library inference: int8 inputs × int8 weights, int accumulator."""

import json
import math
from pathlib import Path

from .features import FEATURE_NAMES, extract
from .suffix import SUFFIX_RULES


def quantize_feature(value, mean, std, input_scale):
    z = (value - mean) / std
    return max(-127, min(127, round(z / input_scale)))


class Int8Model:
    def __init__(self, artifact):
        if artifact.get("format") != "dgaguard-linear-int8-v2":
            raise ValueError("unsupported model format; retrain the model")
        if artifact.get("suffix_rules") != SUFFIX_RULES:
            raise ValueError("model suffix rules do not match this code; retrain the model")
        if artifact.get("features") != list(FEATURE_NAMES):
            raise ValueError("model feature order does not match this code")
        size = len(FEATURE_NAMES)
        if any(len(artifact[key]) != size for key in ("mean", "std", "weights_int8")):
            raise ValueError("model parameter length does not match features")
        if any(value <= 0 for value in artifact["std"]):
            raise ValueError("standard deviations must be positive")
        if artifact["input_scale"] <= 0 or artifact["weight_scale"] <= 0:
            raise ValueError("invalid quantization scale")
        if any(not isinstance(weight, int) or not -127 <= weight <= 127
               for weight in artifact["weights_int8"]):
            raise ValueError("weights must be signed int8")
        self.data = artifact
        self.default_alert_threshold = float(artifact.get("alert_threshold", 0.90))
        self.lexicon = dict(artifact["lexicon"])
        self.lexicon["words"] = set(self.lexicon["words"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def logit(self, vector):
        d = self.data
        q = [quantize_feature(x, m, s, d["input_scale"])
             for x, m, s in zip(vector, d["mean"], d["std"], strict=True)]
        accumulator = sum(a * b for a, b in zip(q, d["weights_int8"], strict=True))
        return accumulator * d["input_scale"] * d["weight_scale"] + d["bias"]

    def score(self, domain):
        vector = extract(domain, self.lexicon)
        if vector is None:
            return None
        logit = self.logit(vector)
        return 1 / (1 + math.exp(-max(-60.0, min(60.0, logit))))


def load_model(path):
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    if str(artifact.get("format", "")).startswith("dgaguard-char-tfidf"):
        from .char_model import CharModel
        return CharModel(artifact)
    return Int8Model(artifact)
