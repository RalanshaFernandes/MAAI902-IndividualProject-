"""Thin client for DFM's disclosure ("Efsah") JSON API.

The API is undocumented; it is what www.dfm.ae itself calls. Every request is
rate-limited (config.REQUEST_DELAY) to stay polite.
"""
import json
import time

import requests

from config import EFSAH_API, HEADERS, REQUEST_DELAY

_session = requests.Session()
_session.headers.update(HEADERS)


def _get(endpoint: str, **params) -> list:
    """GET an Efsah endpoint and return its 'root' list.

    Responses begin with a UTF-8 byte-order mark, so we decode with utf-8-sig
    instead of using response.json().
    """
    base = {"announcement_type": "Disclosure", "types": "", "keyword": "",
            "from": "", "to": "", "cms_resources": "true",
            "h7_datetime_format": "yyyy-MM-dd HH:mm:ss"}
    resp = _session.get(f"{EFSAH_API}/{endpoint}", params={**base, **params}, timeout=60)
    resp.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return json.loads(resp.content.decode("utf-8-sig"))["root"]


def count_disclosures(symbol: str) -> int:
    return _get("efsah_count", lang="en", symbol=symbol)[0]["count"]


def list_disclosures(symbol: str, lang: str = "en", page_size: int = 20) -> list[dict]:
    """Return every disclosure for a symbol in one language, newest first.

    The API silently caps `take` at 20, so we page in steps of 20 and
    de-duplicate by id in case new disclosures shift the pages mid-run.
    """
    total = count_disclosures(symbol)
    records = {}
    for skip in range(0, total, page_size):
        for r in _get("prototype_efsah", lang=lang, symbol=symbol, take=page_size, skip=skip):
            records.setdefault(r["id"], r)
    return list(records.values())
