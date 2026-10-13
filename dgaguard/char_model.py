"""Standard-library scorer that reproduces scikit-learn's character TF-IDF + logistic regression."""

from collections import Counter
import json
import math
from pathlib import Path

from .features import lexical_label
from .suffix import SUFFIX_RULES

FORMAT = "dgaguard-char-tfidf-v2"


def wrap(token):
    """Boundary markers let n-grams tell prefixes and suffixes from interior text."""
    return f"^{token}$"


class CharModel:
    def __init__(self, artifact):
        if artifact.get("format") != FORMAT:
            raise ValueError(f"unsupported model format {artifact.get('format')!r}; retrain the model")
        if artifact.get("suffix_rules") != SUFFIX_RULES:
            raise ValueError(f"model was trained with suffix rules {artifact.get('suffix_rules')!r}, "
                             f"but this code uses {SUFFIX_RULES!r}; retrain the model")
        self.data = artifact
        self.weights = artifact["ngrams"]
        self.bias = float(artifact["bias"])
        self.minimum, self.maximum = artifact["ngram_range"]
        if not 1 <= self.minimum <= self.maximum <= 8:
            raise ValueError("invalid n-gram range")
        self.default_alert_threshold = float(artifact["alert_threshold"])

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def score_token(self, token):
        text = wrap(token)
        counts = Counter(
            text[i:i + n]
            for n in range(self.minimum, self.maximum + 1)
            for i in range(len(text) - n + 1)
        )
        # Each stored entry is [logistic coefficient * idf, idf^2], so this
        # reproduces the L2-normalized TF-IDF dot product without a vector.
        numerator = 0.0
        squared_norm = 0.0
        for gram, count in counts.items():
            entry = self.weights.get(gram)
            if entry:
                numerator += count * entry[0]
                squared_norm += count * count * entry[1]
        logit = self.bias + (numerator / math.sqrt(squared_norm) if squared_norm else 0.0)
        return 1 / (1 + math.exp(-max(-60.0, min(60.0, logit))))

    def score(self, domain):
        token = lexical_label(domain)
        return self.score_token(token) if token is not None else None
