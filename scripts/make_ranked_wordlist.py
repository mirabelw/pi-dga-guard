"""Generate a frequency-ranked English wordlist with optional off-device wordfreq."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/english_ranked.txt")
    parser.add_argument("--count", type=int, default=20000)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("count must be positive")
    try:
        from wordfreq import top_n_list
    except ImportError as exc:
        raise SystemExit("Install requirements-eval.txt on the training machine") from exc
    words = []
    for word in top_n_list("en", max(args.count * 2, 30000), ascii_only=True):
        if word.isascii() and word.isalpha() and 4 <= len(word) <= 24:
            words.append(word)
        if len(words) == args.count:
            break
    if len(words) != args.count:
        raise ValueError("not enough usable frequency-ranked English words")
    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(words) + "\n", encoding="utf-8")
    print(f"Wrote {len(words)} frequency-ranked words to {target}")


if __name__ == "__main__":
    main()
