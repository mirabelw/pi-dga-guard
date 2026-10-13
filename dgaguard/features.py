"""Identical feature extraction on the training machine and the Pi."""

from collections import Counter
import ipaddress
import math
import re

from .suffix import registrable_label


FEATURE_NAMES = (
    "length", "entropy", "digit_ratio", "vowel_ratio",
    "hyphen_ratio", "english_bigram_logp", "longest_word_ratio",
)
ALPHABET = "abcdefghijklmnopqrstuvwxyz"
VALID_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
DEFAULT_LOCAL_SUFFIXES = ("local", "lan", "arpa", "home", "internal", "localdomain", "test")


def normalize_domain(domain, local_suffixes=DEFAULT_LOCAL_SUFFIXES):
    """Return normalized ASCII DNS name or None; reject local/IP/reverse names."""
    if not isinstance(domain, str):
        return None
    name = domain.strip().rstrip(".").lower()
    if not name or len(name) > 253:
        return None
    try:
        name = name.encode("idna").decode("ascii")
    except UnicodeError:
        return None
    if any(name == suffix or name.endswith("." + suffix) for suffix in local_suffixes):
        return None
    try:
        ipaddress.ip_address(name)
        return None
    except ValueError:
        pass
    labels = name.split(".")
    if len(labels) < 2 or any(not VALID_LABEL.fullmatch(label) for label in labels):
        return None
    if labels[-1].isdigit():
        return None
    if any(label.startswith("xn--") for label in labels):
        return None
    return name


def lexical_label(domain):
    """Return the label to score, e.g. 'example' for 'cdn.example.co.uk'."""
    name = normalize_domain(domain)
    return registrable_label(name) if name else None


def build_lexicon(words, limit=20000):
    """Keep the first unique words; frequency-ranked input keeps common words."""
    if limit <= 0:
        raise ValueError("word limit must be positive")
    cleaned = set()
    for word in words:
        word = word.strip().lower()
        if word.isascii() and word.isalpha() and 4 <= len(word) <= 24:
            cleaned.add(word)
        if len(cleaned) >= limit:
            break
    if not cleaned:
        raise ValueError("wordlist needs at least one ASCII word of length 4–24")
    # Smoothing over all letter pairs prevents an unseen pair from reaching -inf.
    counts = Counter(pair for word in cleaned for pair in zip(word, word[1:]))
    total = sum(counts.values())
    logp = {a + b: math.log((counts[a, b] + 1) / (total + 26 * 26))
            for a in ALPHABET for b in ALPHABET}
    return {"words": sorted(cleaned), "bigram_logp": logp}


def extract(domain, lexicon):
    label = lexical_label(domain)
    if label is None:
        return None
    n = len(label)
    counts = Counter(label)
    entropy = -sum((v / n) * math.log2(v / n) for v in counts.values())
    letter_runs = re.findall("[a-z]+", label)
    letters = "".join(letter_runs)
    pairs = [a + b for run in letter_runs for a, b in zip(run, run[1:])]
    # No pairs is less English-like than an unseen English letter pair.
    floor = min(lexicon["bigram_logp"].values())
    bigram = sum(lexicon["bigram_logp"].get(p, floor) for p in pairs) / len(pairs) if pairs else floor
    words = lexicon["words"] if isinstance(lexicon["words"], set) else set(lexicon["words"])
    longest = 0
    # Only substrings of 4–24 chars; a max 63-char DNS label keeps this cheap.
    for run in letter_runs:
        for start in range(len(run)):
            for end in range(start + max(4, longest + 1), min(len(run), start + 24) + 1):
                if run[start:end] in words:
                    longest = end - start
    return [
        float(n), entropy, sum(counts[str(i)] for i in range(10)) / n,
        sum(counts[c] for c in "aeiou") / n,
        counts["-"] / n, bigram, longest / n,
    ]
