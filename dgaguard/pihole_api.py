"""Opt-in Pi-hole v6 exact denylist writer (no direct gravity DB edits)."""

import json
import os
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .features import normalize_domain


class PiholeAPI:
    def __init__(self, base_url, password_env="PIHOLE_APP_PASSWORD"):
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Pi-hole API URL must be HTTP(S) with a hostname")
        if parsed.scheme == "http" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("use local HTTP or HTTPS for Pi-hole API credentials")
        if parsed.username or parsed.password or parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Pi-hole API URL must contain only scheme, host, and optional port")
        self.base_url = base_url.rstrip("/")
        self.password_env = password_env
        self.sid = None

    def _request(self, path, payload=None, sid=None):
        headers = {"Content-Type": "application/json"}
        if sid:
            headers["X-FTL-SID"] = sid
        body = json.dumps(payload).encode() if payload is not None else None
        request = Request(self.base_url + path, data=body, headers=headers,
                          method="POST" if payload is not None else "GET")
        with urlopen(request, timeout=5) as response:
            data = response.read()
            return json.loads(data) if data else {}

    def authenticate(self):
        password = os.environ.get(self.password_env)
        if not password:
            raise RuntimeError(f"set {self.password_env} to a Pi-hole v6 application password")
        response = self._request("/api/auth", {"password": password})
        self.sid = response["session"]["sid"]

    def deny_exact(self, domain):
        if normalize_domain(domain) != domain:
            raise ValueError("invalid exact domain")
        if self.sid is None:
            self.authenticate()
        payload = {"domain": domain, "comment": "DGA Guard: review and remove if false positive"}
        for attempt in range(2):
            try:
                self._request("/api/domains/deny/exact", payload, self.sid)
                return
            except HTTPError as exc:
                if exc.code == 401 and attempt == 0:
                    self.authenticate()
                    continue
                raise
