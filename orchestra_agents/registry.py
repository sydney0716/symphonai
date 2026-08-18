"""In-memory registry mapping a provider name to a SubagentProvider instance.

The registry never constructs vendor CLI/API clients itself and has no
import-time side effects; it only holds whatever `SubagentProvider`
instances a caller explicitly registers.
"""

from __future__ import annotations

from orchestra_agents.base import SubagentProvider

_registry: dict[str, SubagentProvider] = {}


def register_provider(provider: SubagentProvider, *, overwrite: bool = False) -> None:
    """Register `provider` under `provider.name`.

    Raises ValueError if a provider is already registered under that name,
    unless `overwrite=True`.
    """
    if provider.name in _registry and not overwrite:
        raise ValueError(f"provider already registered: {provider.name!r}")
    _registry[provider.name] = provider


def get_provider(name: str) -> SubagentProvider:
    """Look up a previously registered provider by name.

    Raises KeyError if no provider is registered under that name.
    """
    try:
        return _registry[name]
    except KeyError as exc:
        raise KeyError(f"no provider registered under {name!r}") from exc


def list_providers() -> list[str]:
    """Return the names of all currently registered providers, sorted."""
    return sorted(_registry)


def clear_registry() -> None:
    """Remove all registered providers. Mainly useful for tests."""
    _registry.clear()
