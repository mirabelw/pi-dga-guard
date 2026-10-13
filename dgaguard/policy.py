"""Optional popular-domain exclusion for Pi-side scoring."""

from pathlib import Path

from .features import normalize_domain


def load_allowlist(path):
    """Accept domains one per line, or Tranco-style rank,domain CSV."""
    domains = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(",")
        value = fields[1].strip() if len(fields) == 2 and fields[0].strip().isdigit() else fields[0].strip()
        domain = normalize_domain(value)
        if domain:
            domains.add(domain)
    if not domains:
        raise ValueError("allowlist has no valid domains")
    return domains


def is_allowlisted(domain, allowlist):
    """Only match at a label boundary: example.com and *.example.com."""
    labels = domain.split(".")
    return any(".".join(labels[index:]) in allowlist for index in range(len(labels) - 1))
