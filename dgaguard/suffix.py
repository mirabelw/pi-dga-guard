"""Public Suffix List matcher: the ICANN section plus dynamic-DNS providers.

Names under other private suffixes (cloudfront.net, github.io, ...) are scored on the
provider's label, since their subdomains are often random IDs that would be wrongly flagged.
DDNS labels stay scoreable because DGAs such as corebot use them.
"""

from functools import lru_cache
from pathlib import Path
import re

SUFFIX_RULES = "psl-icann+ddns-v1"
PSL_PATH = Path(__file__).resolve().parents[1] / "data" / "public_suffix_list.dat"
# Organization names exactly as they appear in the PSL private-section headers.
DDNS_PROVIDERS = frozenset({
    "ddnss.de", "deSEC", "dnsHome", "DuckDNS", "dy.fi", "DynDNS.com",
    "Dynu.com", "dynv6", "IPv64.net", "No-IP.com", "Now-DNS", "nsupdate.info",
})
_HEADER = re.compile(r"^// (.+?) : https?://")


@lru_cache(maxsize=1)
def rules():
    exact, wildcard, exception = set(), set(), set()
    section, organization, found = None, None, set()
    for raw in PSL_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if "===BEGIN ICANN DOMAINS===" in line:
            section = "icann"
            continue
        if "===BEGIN PRIVATE DOMAINS===" in line:
            section = "private"
            continue
        if "===END" in line:
            section = None
            continue
        header = _HEADER.match(line)
        if header:
            organization = header.group(1)
            continue
        if section is None or not line or line.startswith("//"):
            continue
        if section == "private":
            if organization not in DDNS_PROVIDERS:
                continue
            found.add(organization)
        item = line.lower()
        # IDN/punycode names are rejected before scoring, so non-ASCII rules
        # can never match.
        if not item.isascii():
            continue
        if item.startswith("!"):
            exception.add(item[1:])
        elif item.startswith("*."):
            wildcard.add(item[2:])
        else:
            exact.add(item)
    missing = DDNS_PROVIDERS - found
    if not exact or missing:
        raise ValueError(f"PSL at {PSL_PATH} is missing expected sections: {sorted(missing)}")
    return exact, wildcard, exception


def public_suffix_length(labels):
    """Number of trailing labels forming the public suffix (PSL algorithm)."""
    exact, wildcard, exceptions = rules()
    best = 1  # PSL prevailing rule: '*' if no rule matches.
    for index in range(len(labels)):
        suffix = ".".join(labels[index:])
        remaining = len(labels) - index
        if suffix in exceptions:
            return max(1, remaining - 1)
        if suffix in exact:
            best = max(best, remaining)
        if index > 0 and suffix in wildcard:
            best = max(best, remaining + 1)
    return best


def registrable_label(name):
    labels = name.split(".")
    best = public_suffix_length(labels)
    return labels[-best - 1] if len(labels) > best else None


def public_suffix(name):
    labels = name.split(".")
    best = public_suffix_length(labels)
    return ".".join(labels[-best:]) if len(labels) > best else None
