"""Negative-result experiment (see results/further_experiments.md). Usage: DATA_DIR=data python scripts/extra/mlp.py <family|known> <hidden> <max_features> <epochs>"""
import csv, json, sys, time
import os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..')); DATA = os.environ.get('DATA_DIR', 'data')
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score
from scripts.experiments import load, recall_at_fpr
mode, hidden, feats, epochs = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
tr, va = load(f'{DATA}/{mode}_train.csv'), load(f'{DATA}/{mode}_validation.csv')
vec = TfidfVectorizer(analyzer='char', ngram_range=(1, 6), lowercase=False, min_df=3,
                      max_features=feats, dtype=np.float32)
Xtr = vec.fit_transform([f"^{r['token']}$" for r in tr]); ytr = np.array([r['label'] for r in tr])
Xva = vec.transform([f"^{r['token']}$" for r in va]); yva = [r['label'] for r in va]
clf = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
                    max_iter=1, warm_start=True, random_state=42)
log = open(f'results/extra_mlp_{mode}_{hidden}_{feats}.log', 'w')
for epoch in range(epochs):
    t = time.time(); clf.fit(Xtr, ytr)
    s = clf.predict_proba(Xva)[:, 1]
    print(f'{mode} h={hidden} f={feats} epoch {epoch+1} R@0.1%={recall_at_fpr(yva, s):.3f} '
          f'AP={average_precision_score(yva, s):.4f} ({time.time()-t:.0f}s)', file=log, flush=True)
