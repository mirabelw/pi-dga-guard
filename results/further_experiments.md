# Further accuracy experiments (validation only, all negative)

After the selection in `model_selection.json`, three more ideas were tested against
the shipped configuration. Metric: recall at 0.1% benign false positives, with AP
in parentheses. Test data was not used.

| Idea | Unseen-family validation | Known-family validation | Verdict |
| --- | --- | --- | --- |
| Shipped logistic regression | 0.270 (0.849) | 0.605 (0.990) | reference |
| Logit ensemble with the 7-feature model (weight 0.1–1.5) | 0.268–0.272 (0.844–0.851) | 0.579–0.607 (0.984–0.990) | no gain |
| MLP, 64 hidden units, 40k features, 4–6 epochs | 0.231–0.267 (0.829–0.838) | 0.612–0.631 (0.987–0.989) | worse on unseen families; known-family gain within noise; about 10x slower on the Pi |
| Drop the most DGA-like 0.5% / 1% of benign training labels (2-fold cross-fit) | 0.236 / 0.252 (0.844 / 0.849) | not run | worse; also removes hard legitimate names |

None of these beat the shipped model. The next gain likely needs new information
(query context) rather than a different lexical model.
Scripts: `scripts/extra/mlp.py` and `scripts/extra/clean.py`. The ensemble
combined the two trained models' logits on the validation split.
