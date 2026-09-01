"""Reader for the shipped web-fetch domain table."""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def preapproved_domains() -> tuple[str, ...]:
    """The shipped preapproved list, read once and cached."""

    try:
        payload = json.loads(
            resources.files("symphonai_api.data")
            .joinpath("web_domains.json")
            .read_text(encoding="utf-8")
        )
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            return ()
        domains = payload.get("preapproved")
        if not isinstance(domains, list) or not all(
            isinstance(domain, str) and domain for domain in domains
        ):
            return ()
        return tuple(domain.casefold().rstrip(".") for domain in domains)
    except (OSError, TypeError, ValueError, UnicodeError):
        return ()
