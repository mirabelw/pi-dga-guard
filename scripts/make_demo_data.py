"""Synthetic smoke-test CSV only. Do not use its scores as a security benchmark."""

import csv
from pathlib import Path
import random
import string
import sys


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/demo.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)
    words = [
        "garden", "market", "music", "coffee", "river", "cloud", "school",
        "travel", "orange", "forest", "winter", "summer", "science", "sports",
        "studio", "bright", "health", "family", "design", "friend",
    ]
    rows = []
    for i in range(200):
        benign = rng.choice(words) + rng.choice(words) + str(i % 10)
        random_label = "".join(rng.choices(string.ascii_lowercase + string.digits, k=rng.randint(11, 19)))
        rows.extend([(benign + ".example", 0), (random_label + ".example", 1)])
    rng.shuffle(rows)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["domain", "label"])
        writer.writerows(rows)
    print(f"Wrote {len(rows)} synthetic rows to {path}; never deploy this demo model")


if __name__ == "__main__":
    main()
