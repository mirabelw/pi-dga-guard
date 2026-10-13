import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
from dgaguard.features import FEATURE_NAMES, build_lexicon, extract, lexical_label, normalize_domain
from dgaguard.model import Int8Model
from dgaguard.char_model import FORMAT as CHAR_FORMAT, CharModel
from dgaguard.suffix import SUFFIX_RULES
from dgaguard.monitor import poll_once
from dgaguard.pihole_api import PiholeAPI
from dgaguard.policy import is_allowlisted, load_allowlist


class StubModel:
    def score(self, domain):
        return 0.999 if domain.startswith("qxz") else 0.01


class StubAPI:
    def __init__(self):
        self.denied = []

    def deny_exact(self, domain):
        self.denied.append(domain)


class GuardTests(unittest.TestCase):
    def test_char_model_matches_sklearn_training_scores(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("training dependencies are not installed")
        from dgaguard.train_char import train
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_file = root / "domains.csv"
            with csv_file.open("w", encoding="utf-8") as stream:
                stream.write("domain,label\n")
                for i in range(80):
                    stream.write(f"gardenmarket{i}.example,benign\n")
                    stream.write(f"qxzv{i}91qkpt.example,dga\n")
            model_file, report_file = root / "model.json", root / "report.json"
            report = train(csv_file, model_file, report_file, max_features=2000)
            model = CharModel.load(model_file)
            self.assertLess(report["portable_score_max_abs_difference"], 1e-5)
            self.assertTrue(0 <= model.score("qxzv9391qkpt.example") <= 1)
            self.assertEqual(report["test"]["count"] + report["validation"]["count"]
                             + report["training_count"], 160)

    def test_char_training_rejects_registrable_label_leakage(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("training dependencies are not installed")
        from dgaguard.train_char import train
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            training, validation = root / "train.csv", root / "validation.csv"
            with training.open("w", encoding="utf-8") as stream:
                stream.write("domain,label\n")
                for i in range(20):
                    stream.write(f"gardenmarket{i}.com,benign\nqxzv{i}91qkpt.com,dga\n")
            with validation.open("w", encoding="utf-8") as stream:
                stream.write("domain,label\n")
                for i in range(19):
                    stream.write(f"someword{i}.org,benign\nqwdkj{i}91.org,dga\n")
                stream.write("cdn.gardenmarket0.co.uk,benign\n")
            with self.assertRaisesRegex(ValueError, "lexical token overlap"):
                train(training, root / "model.json", root / "report.json",
                      validation_csv=validation)

    def test_char_model_rejects_mismatched_suffix_rules(self):
        artifact = {"format": CHAR_FORMAT, "suffix_rules": "psl-icann+private",
                    "ngram_range": [1, 6], "bias": 0.0, "ngrams": {}, "alert_threshold": 0.99}
        with self.assertRaisesRegex(ValueError, "retrain"):
            CharModel(artifact)
        artifact["suffix_rules"] = SUFFIX_RULES
        self.assertAlmostEqual(CharModel(artifact).score("example.com"), 0.5)

    def test_feature_normalization_and_extraction(self):
        lexicon = build_lexicon(["garden", "market", "school"])
        self.assertEqual(normalize_domain("Market.EXAMPLE."), "market.example")
        self.assertIsNone(normalize_domain("printer.local"))
        self.assertIsNone(normalize_domain("127.0.0.1"))
        self.assertIsNone(normalize_domain("invalid_label.com"))
        self.assertIsNone(normalize_domain("printer.localdomain"))
        self.assertIsNone(normalize_domain("example.test"))
        self.assertIsNone(normalize_domain("xn--bcher-kva.com"))
        self.assertIsNone(normalize_domain("printer.corp", local_suffixes=("corp",)))
        features = extract("garden.example", lexicon)
        self.assertEqual(len(features), 7)
        self.assertEqual(features[0], 6.0)
        self.assertEqual(features[-1], 1.0)
        self.assertEqual(len(FEATURE_NAMES), 7)
        self.assertEqual(lexical_label("foo.example.co.uk"), "example")
        # Documented trade-off: non-DDNS private suffixes are not boundaries.
        self.assertEqual(lexical_label("plasticbags.sa.com"), "sa")
        # Dynamic-DNS labels are claimable by anyone, so they stay scoreable...
        self.assertEqual(lexical_label("randomname.ddns.net"), "randomname")
        self.assertEqual(lexical_label("randomname.duckdns.org"), "randomname")
        # ...but provider-assigned CDN/hosting IDs collapse to the provider name.
        self.assertEqual(lexical_label("d3k2j8f9s1a0bx.cloudfront.net"), "cloudfront")
        self.assertEqual(lexical_label("zq9x8w7v.github.io"), "github")
        # PSL wildcard and exception rules.
        self.assertEqual(lexical_label("a.b.kawasaki.jp"), "a")
        self.assertEqual(lexical_label("city.kawasaki.jp"), "city")
        self.assertIsNone(lexical_label("co.uk"))

    def test_wordlist_cap_and_no_fake_letter_pairs(self):
        words = ["aaaa", "bbbb", "cccc", "dddd"]
        lexicon = build_lexicon(words, limit=2)
        self.assertEqual(lexicon["words"], ["aaaa", "bbbb"])
        floor = min(lexicon["bigram_logp"].values())
        self.assertEqual(extract("12345678.example", lexicon)[-2], floor)
        paired = (lexicon["bigram_logp"]["ab"] + lexicon["bigram_logp"]["bc"]
                  + lexicon["bigram_logp"]["de"] + lexicon["bigram_logp"]["ef"]) / 4
        self.assertAlmostEqual(extract("abc123def.example", lexicon)[-2], paired)
        self.assertEqual(extract("abc123def.example", build_lexicon(["abcdef"]))[-1], 0)

    def test_allowlist_exact_and_subdomains(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "top.csv"
            path.write_text("1,example.com\n2,google.com\n", encoding="utf-8")
            names = load_allowlist(path)
        self.assertTrue(is_allowlisted("cdn.example.com", names))
        self.assertFalse(is_allowlisted("fakeexample.com", names))

    def test_int8_accumulation(self):
        lexicon = build_lexicon(["garden"])
        data = {"format": "dgaguard-linear-int8-v2", "suffix_rules": SUFFIX_RULES,
                "features": list(FEATURE_NAMES),
                "mean": [0.0] * 7, "std": [1.0] * 7,
                "input_scale": 1.0, "weight_scale": 0.1,
                "weights_int8": [1] + [0] * 6, "bias": 0.0,
                "lexicon": lexicon}
        model = Int8Model(data)
        self.assertAlmostEqual(model.logit([6.0] + [0.0] * 6), 0.6)

    def test_monitor_cursor_cooldown_and_opt_in_block(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "queries.db"
            state = root / "cursor.json"
            alerts = root / "alerts.jsonl"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE queries (id INTEGER PRIMARY KEY, timestamp INTEGER, domain TEXT, status INTEGER)")
                conn.executemany("INSERT INTO queries VALUES (?, ?, ?, ?)", [
                    (1, 10, "qxzbad.example", 2),
                    (2, 11, "qxzblocked.example", 5),
                    (3, 12, "qxzbad.example", 3),
                ])
            api = StubAPI()
            processed = poll_once(StubModel(), db, state, alerts, backfill=True, autoblock=api)
            self.assertEqual(processed, 3)
            self.assertEqual(api.denied, ["qxzbad.example"])
            events = [json.loads(line) for line in alerts.read_text().splitlines()]
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]["action"], "blocked")
            self.assertEqual(poll_once(StubModel(), db, state, alerts, autoblock=api), 0)
            with sqlite3.connect(db) as conn:
                conn.execute("INSERT INTO queries VALUES (4, 13, 'qxzbad.example', 2)")
            self.assertEqual(poll_once(StubModel(), db, state, alerts, autoblock=api), 1)
            self.assertEqual(api.denied, ["qxzbad.example"])

    def test_api_auth_and_exact_deny_request(self):
        api = PiholeAPI("http://127.0.0.1")
        calls = []
        def fake_request(path, payload=None, sid=None):
            calls.append((path, payload, sid))
            if path == "/api/auth":
                return {"session": {"sid": "temporary"}}
            return {}
        with patch.dict("os.environ", {"PIHOLE_APP_PASSWORD": "private"}):
            with patch.object(api, "_request", side_effect=fake_request):
                api.deny_exact("qxzbad.example")
        self.assertEqual(calls[0][0], "/api/auth")
        self.assertEqual(calls[1], ("/api/domains/deny/exact",
                                   {"domain": "qxzbad.example", "comment": "DGA Guard: review and remove if false positive"},
                                   "temporary"))

    def test_api_rejects_credential_leak_and_missing_password(self):
        for url in ("http://localhost.attacker.example", "http://127.0.0.1.attacker.example",
                    "http://user:pass@127.0.0.1"):
            with self.assertRaises(ValueError):
                PiholeAPI(url)
        api = PiholeAPI("http://127.0.0.1")
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(RuntimeError):
                api.authenticate()

    def test_training_pipeline_exports_matching_feature_order(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("training dependency not installed on this machine")
        from dgaguard.train import train
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv_path, wordlist = root / "input.csv", root / "words.txt"
            model_path, report_path = root / "model.json", root / "report.json"
            wordlist.write_text("garden\nmarket\nriver\n", encoding="utf-8")
            with csv_path.open("w", encoding="utf-8") as stream:
                stream.write("domain,label\n")
                for i in range(40):
                    stream.write(f"gardenmarket{i}.example,benign\n")
                    stream.write(f"qxzv{i}91qkpt.example,dga\n")
            report = train(csv_path, model_path, report_path, wordlist, max_words=2)
            model = Int8Model.load(model_path)
            self.assertEqual(len(model.data["weights_int8"]), 7)
            self.assertEqual(report["parameters"]["wordlist_size"], 2)
            self.assertIn("false_positive_domains", report["validation"])
            self.assertEqual(model.default_alert_threshold, report["validation"]["threshold"])

    def test_first_start_tails_and_does_not_replay_old_queries(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root / "queries.db"
            state = root / "cursor.json"
            alerts = root / "alerts.jsonl"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE queries (id INTEGER PRIMARY KEY, timestamp INTEGER, domain TEXT, status INTEGER)")
                conn.execute("INSERT INTO queries VALUES (1, 10, 'qxzold.example', 2)")
            self.assertEqual(poll_once(StubModel(), db, state, alerts), 0)
            self.assertFalse(alerts.exists())
            with sqlite3.connect(db) as conn:
                conn.execute("INSERT INTO queries VALUES (2, 11, 'qxznew.example', 2)")
            self.assertEqual(poll_once(StubModel(), db, state, alerts), 1)
            self.assertEqual(json.loads(alerts.read_text())["domain"], "qxznew.example")


if __name__ == "__main__":
    unittest.main()
