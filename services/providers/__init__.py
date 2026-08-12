"""Live data providers: public endpoints that supply real player data.

Each provider fetches and shapes exactly one source. None of them join, blend, or
interpret — that is :mod:`services.providers.resolver`'s job. Every provider
returns a :class:`~services.providers.base.ProviderResult` and never raises, so a
dead source degrades the board rather than breaking the app.

Sources shipped here are public JSON endpoints that answer without
authentication. No provider parses HTML or circumvents a login or paywall.
"""

from __future__ import annotations

from typing import Any

from services.providers.base import (
    DataProvider,
    ProviderResult,
    cache_entries,
    clear_cache,
    fetch_bytes,
    fetch_json,
)
from services.providers.espn import ESPNProvider
from services.providers.ffcalculator import FFCalculatorProvider
from services.providers.sleeper import SleeperProvider
from services.providers.yahoo import YahooProvider

# Order matters for display and for the resolver's precedence: Sleeper is the
# identity spine, FFC is the primary ADP source (it is the only one publishing a
# spread), and ESPN/Yahoo are corroborating platform views.
PROVIDERS: tuple[type, ...] = (
    SleeperProvider,
    FFCalculatorProvider,
    ESPNProvider,
    YahooProvider,
)


def all_providers() -> list[Any]:
    """One instance of every shipped provider, in precedence order."""
    return [provider() for provider in PROVIDERS]


def provider_by_key(key: str) -> Any | None:
    """Look up a provider instance by its ``key``, or ``None``."""
    for provider in PROVIDERS:
        if provider.key == key:
            return provider()
    return None


__all__ = [
    "DataProvider",
    "ProviderResult",
    "SleeperProvider",
    "FFCalculatorProvider",
    "ESPNProvider",
    "YahooProvider",
    "PROVIDERS",
    "all_providers",
    "provider_by_key",
    "cache_entries",
    "clear_cache",
    "fetch_bytes",
    "fetch_json",
]
