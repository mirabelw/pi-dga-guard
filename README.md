# Pi DGA Guard

Pi DGA Guard watches the domain lookups on your home network and warns you when a device looks up a name that seems machine-generated, a common sign of malware.

It runs on a Raspberry Pi next to [Pi-hole](https://pi-hole.net/). Many kinds of malware find their control server by generating thousands of random-looking domain names (a **DGA**, domain generation algorithm), like `xjwqkzvbnrtplm.com`. Pi-hole's blocklists only stop names already known to be bad. Pi DGA Guard catches new ones by scoring how "generated" each name looks, using a small model trained on known malware domains.

## Features

- **Character n-gram model.** Logistic regression over TF-IDF weights of 1–6-character substrings, trained with scikit-learn on a desktop.
- **No dependencies on the Pi.** Scoring uses only Python's standard library and matches scikit-learn to within 4×10⁻⁷.
- **Scores the right part of the name.** It uses the Public Suffix List: `mail.example.co.uk` → `example`. Dynamic-DNS services such as `ddns.net` count as suffixes, because DGAs use them. CDN hosts do not, so `d3k2j8f9s1a0bx.cloudfront.net` → `cloudfront` and doesn't cause false alarms.
- **Built for precision.** The alert threshold is set so only about 0.1% of legitimate domains are flagged. That means it misses some DGA domains, on purpose: a false alarm is a nuisance in the log, but with auto-block it would silently break a real site for everyone on the network. Malware usually tries many generated names, so catching a fraction can still reveal an infected device.
- **Safe by default.** It reads Pi-hole's query database read-only, writes alerts to a log, and only blocks domains if you opt in (through Pi-hole's v6 API).
- **Evaluated honestly.** Label-disjoint splits, a separate test on malware families never seen in training, thresholds fixed before any test data is used, and documented negative results.

## Results

Thresholds favor precision: about 0.1% of legitimate domains are flagged, at the cost of catching fewer DGA domains.

| Test | DGA domains caught | Legitimate domains flagged |
| --- | --- | --- |
| New domains from 25 known malware families | 60% (7,959 / 13,227) | 9 / 12,959 |
| 3 malware families never seen in training | 33% (4,014 / 12,047) | 1 / 12,729 |

The main weakness is DGAs built from real English words (e.g. matsnu: under 1% caught). A model that judges one name at a time can't separate these from real domains. Scores are not probabilities, and results on a real home network will differ. Details: [`results/`](results/).

## Usage

### On the Pi

Requires Pi-hole v6 and Python 3.10+. There is nothing to `pip install`.

```bash
sudo git clone https://github.com/YOUR-USERNAME/pi-dga-guard.git /opt/pi-dga-guard
cd /opt/pi-dga-guard
python3 -m dgaguard score example.org      # score one domain
python3 -m dgaguard monitor --once         # one polling pass (sets the starting point)
```

To run it continuously as a service (uses a throwaway unprivileged user in the `pihole` group):

```bash
sudo install -m 0644 deploy/pi-dga-guard.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now pi-dga-guard
journalctl -u pi-dga-guard -f              # alerts appear here
```

Alerts are also saved to `/var/lib/pi-dga-guard/alerts.jsonl`. These useful flags go on `monitor`:

| Flag | What it does |
| --- | --- |
| `--allowlist FILE` | Skip trusted domains, e.g. a Tranco top-sites CSV |
| `--alert-threshold X` | Override the default threshold |
| `--auto-block --block-threshold X` | Add flagged domains to Pi-hole's denylist; needs `PIHOLE_APP_PASSWORD` set |

Auto-blocking is not recommended until you have reviewed a week or more of alerts from your own network.

### Training (desktop)

```bash
pip install -r requirements-eval.txt
python -m unittest discover -s tests
curl -L -o data/dga_domains_full.csv \
  https://raw.githubusercontent.com/chrmor/DGA_domains_dataset/master/dga_domains_full.csv
python scripts/prepare_dataset.py --source data/dga_domains_full.csv --mode known --full-train
python -m dgaguard train --csv data/known_train.csv --validation-csv data/known_validation.csv \
  --test-csv data/known_test.csv --model models/dga_model.json --report results/known_family_report.json
```

Use `--mode family` for the unseen-family experiment, and `--model-type linear` for the 7-feature baseline.

## Repository organization

| Path | Contents |
| --- | --- |
| `dgaguard/` | The package: scoring (`char_model.py`, `suffix.py`), training (`train_char.py`, baseline `train.py`), Pi-hole monitor and API client |
| `models/` | The trained model shipped for the Pi |
| `scripts/` | Dataset preparation, model selection, final comparison, benchmark, and extra experiments (`extra/`) |
| `results/` | Evaluation reports, model-selection logs, benchmark, negative results |
| `deploy/` | systemd service file |
| `tests/` | Unit tests (also run by GitHub Actions) |
| `data/` | Public Suffix List snapshot and split manifests (the dataset itself is downloaded) |

## Next steps

- [ ] **Quantize the model.** Store the 150,000 float weights as int8 with a scale factor, then measure memory, speed, and how much alerts change. (The 7-feature baseline is int8, but it isn't the deployed model.)
- [ ] **Try a larger neural network, quantized as needed.** Train a bigger or sequence-based model and compare it with logistic regression on the same splits, using int8 quantization if it is too big or slow for the Pi. A small neural network didn't beat logistic regression, but larger architectures are untested, and quantization would keep a bigger model small and fast enough for the Pi.
- [ ] **Test on a real Pi.** Run `scripts/benchmark.py` on the Pi, then run in alert-only mode for a week and review every alert. All results so far come from a desktop and a public dataset. Real traffic will show the true false-alarm rate, which decides whether auto-block is ever safe to turn on.
- [ ] **Add context signals.** Flag a device that suddenly looks up many domains that don't exist. DGAs built from real words look like normal names, so the name alone can't catch them, but the behavior can. The lexical model appears saturated on this dataset (see [`results/further_experiments.md`](results/further_experiments.md)).
- [ ] **Alert notifications or a web page.** Push alerts to a phone or list recent ones on a page. Right now you have to read the system log to see alerts.
- [ ] **Refresh the Public Suffix List** periodically, and retrain afterward. New suffixes change which part of a name gets scored, and the model must match.

MIT license. The bundled Public Suffix List is MPL-2.0.
