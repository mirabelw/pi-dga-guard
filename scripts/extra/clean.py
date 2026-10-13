"""Negative-result experiment (see results/further_experiments.md). Usage: DATA_DIR=data python scripts/extra/clean.py <family|known>"""
import sys, time, json
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); DATA = os.environ.get('DATA_DIR', 'data')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from scripts.experiments import load, recall_at_fpr
mode = sys.argv[1]; out = open(f'results/extra_clean_{mode}.log', 'w')
tr, va = load(f'{DATA}/{mode}_train.csv'), load(f'{DATA}/{mode}_validation.csv')
def fit(rows):
    v = TfidfVectorizer(analyzer='char', ngram_range=(1, 6), lowercase=False, min_df=3,
                        max_features=150000, dtype=np.float32)
    X = v.fit_transform([f"^{r['token']}$" for r in rows])
    c = LogisticRegression(C=20, class_weight='balanced', max_iter=2000, random_state=42)
    c.fit(X, [r['label'] for r in rows]); return v, c
def score(model, rows):
    v, c = model; return c.predict_proba(v.transform([f"^{r['token']}$" for r in rows]))[:, 1]
rng = np.random.RandomState(0); fold = rng.randint(0, 2, len(tr))
oof = np.zeros(len(tr))
for k in (0, 1):
    t = time.time(); m = fit([r for r, f in zip(tr, fold) if f != k])
    idx = np.where(fold == k)[0]; oof[idx] = score(m, [tr[i] for i in idx])
    print(f'fold {k} {time.time()-t:.0f}s', file=out, flush=True)
benign = np.array([r['label'] == 0 for r in tr])
yva = [r['label'] for r in va]
for pct in (0.5, 1.0, 2.0):
    cut = np.percentile(oof[benign], 100 - pct)
    kept = [r for r, s, b in zip(tr, oof, benign) if not (b and s >= cut)]
    dropped = [r['domain'] for r, s, b in zip(tr, oof, benign) if b and s >= cut]
    t = time.time(); s = score(fit(kept), va)
    print(f'{mode} drop top {pct}% benign ({len(tr)-len(kept)}) R@0.1%={recall_at_fpr(yva, s):.3f} '
          f'AP={average_precision_score(yva, s):.4f} ({time.time()-t:.0f}s) e.g. {dropped[:6]}', file=out, flush=True)
