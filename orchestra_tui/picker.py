"""Provider and model picker screens for the Orchestra Textual TUI."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from textual import work
from textual.app import ComposeResult
from textual.screen import Screen
from textual.widgets import Button, Checkbox, Footer, Input, Select, Static

from orchestra_api.model_discovery import list_models
from orchestra_api.provider_catalog import build_catalog_provider, catalog_keys
from orchestra_api.providers.anthropic_provider import AnthropicProvider
from orchestra_api.providers.base import ModelProvider, ProviderError
from orchestra_api.providers.gemini_provider import GeminiProvider
from orchestra_api.providers.openai_provider import OpenAIProvider

RoleName = Literal["leader", "subagent"]

NATIVE_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}
NATIVE_CHOICES = list(NATIVE_PROVIDERS)
CATALOG_CHOICES = catalog_keys()
PICKER_PROVIDER_CHOICES = NATIVE_CHOICES + CATALOG_CHOICES


@dataclass(frozen=True)
class ProviderPickerResult:
    """The concrete providers selected by the picker."""

    leader_provider: ModelProvider
    subagent_provider: ModelProvider


@dataclass
class _RoleState:
    provider_name: str | None = None
    default_model: str = ""
    selected_model: str = ""
    include_all: bool = False
    loading: bool = False
    load_generation: int = 0
    models: list[str] = field(default_factory=list)


def build_picker_provider(provider_name: str, model: str | None = None) -> ModelProvider:
    """Build a real provider offered by the picker.

    The fake provider is deliberately not handled here; it remains available
    only to tests and explicit launcher escape hatches.
    """

    name = provider_name.strip().lower()
    if name in NATIVE_PROVIDERS:
        provider_cls = NATIVE_PROVIDERS[name]
        return provider_cls(model=model) if model else provider_cls()
    if name in CATALOG_CHOICES:
        return build_catalog_provider(name, model=model)
    raise ValueError(f"unknown provider {provider_name!r}")


def provider_summary(provider: ModelProvider) -> str:
    """Human-readable provider/model summary without exposing configuration secrets."""

    model = getattr(provider, "model", None)
    return f"{provider.name} ({model})" if model else provider.name


def uses_real_provider(provider: ModelProvider) -> bool:
    """Return True for providers that may perform real network/API work."""

    return provider.name != "fake"


def _default_model(provider: ModelProvider) -> str:
    model = getattr(provider, "model", "")
    return str(model).strip()


def _one_line(message: object) -> str:
    return " ".join(str(message).split())


def _provider_options() -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for index, provider_name in enumerate(PICKER_PROVIDER_CHOICES, start=1):
        provider_kind = "native" if provider_name in NATIVE_PROVIDERS else "catalog"
        options.append((f"{index}. {provider_name} ({provider_kind})", provider_name))
    return options


class ProviderPickerScreen(Screen[ProviderPickerResult | None]):
    """Pick leader and subagent providers/models before chat starts."""

    BINDINGS = [("escape", "quit", "Quit")]

    def __init__(
        self,
        *,
        leader_provider: ModelProvider | None = None,
        subagent_provider: ModelProvider | None = None,
        confirm_real_providers: bool = False,
        on_real_providers_confirmed: Callable[[], None] | None = None,
        cancel_label: str = "Quit",
    ) -> None:
        super().__init__()
        self._roles: dict[RoleName, _RoleState] = {
            "leader": _RoleState(),
            "subagent": _RoleState(),
        }
        self._initial_providers = {
            "leader": leader_provider,
            "subagent": subagent_provider,
        }
        self._confirm_before_discovery = confirm_real_providers
        self._discovery_confirmed = not confirm_real_providers
        self._on_real_providers_confirmed = on_real_providers_confirmed
        self._awaiting_confirmation = False
        self._pending_provider_changes: dict[RoleName, str] = {}
        self._cancel_label = cancel_label

    def compose(self) -> ComposeResult:
        yield Static("Orchestra provider setup", id="picker-title")
        yield Static(
            "Choose a leader provider/model and one shared subagent provider/model.",
            id="picker-help",
        )
        yield from self._compose_role("leader", "Leader")
        yield from self._compose_role("subagent", "Subagent")
        yield Static("", id="picker-error")
        yield Button("Start chat", id="start-chat", variant="primary", disabled=True)
        yield Button(self._cancel_label, id="quit-picker")
        yield Footer()

    def _compose_role(self, role: RoleName, label: str) -> ComposeResult:
        yield Static(label, classes="picker-role-title")
        yield Select(
            _provider_options(),
            prompt=f"Choose {label.lower()} provider",
            allow_blank=True,
            id=f"{role}-provider",
        )
        yield Checkbox(
            "Show all models",
            id=f"{role}-include-all",
            disabled=True,
        )
        yield Select(
            [],
            prompt="Models load after provider selection",
            allow_blank=True,
            id=f"{role}-model",
            disabled=True,
        )
        yield Input(
            placeholder="Manual model name",
            id=f"{role}-manual-model",
            disabled=True,
        )
        yield Static(f"Select a {label.lower()} provider.", id=f"{role}-model-status")

    def on_mount(self) -> None:
        for role, provider in self._initial_providers.items():
            if provider is not None and provider.name in PICKER_PROVIDER_CHOICES:
                self.query_one(f"#{role}-provider", Select).value = provider.name
        self.query_one("#leader-provider", Select).focus()

    def action_quit(self) -> None:
        self.dismiss(None)

    def on_select_changed(self, event: Select.Changed) -> None:
        widget_id = event.select.id or ""
        if widget_id.endswith("-provider"):
            role = self._role_from_id(widget_id, "-provider")
            if role is None:
                return
            if not isinstance(event.value, str):
                self._reset_role(role)
                return
            self._provider_changed(role, event.value)
            return

        if widget_id.endswith("-model"):
            role = self._role_from_id(widget_id, "-model")
            if role is None or not isinstance(event.value, str):
                return
            self._roles[role].selected_model = event.value
            self.query_one(f"#{role}-manual-model", Input).value = event.value
            self._update_start_button()

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        widget_id = event.checkbox.id or ""
        if not widget_id.endswith("-include-all"):
            return
        role = self._role_from_id(widget_id, "-include-all")
        if role is None:
            return
        state = self._roles[role]
        state.include_all = event.value
        if state.provider_name:
            self._start_model_load(role, state.provider_name)

    def on_input_changed(self, event: Input.Changed) -> None:
        widget_id = event.input.id or ""
        if not widget_id.endswith("-manual-model"):
            return
        role = self._role_from_id(widget_id, "-manual-model")
        if role is None:
            return
        self._roles[role].selected_model = event.value.strip()
        self._update_start_button()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id == "quit-picker":
            self.dismiss(None)
            return
        if button_id != "start-chat":
            return
        result = self._build_result()
        if result is not None:
            self.dismiss(result)

    def _provider_changed(self, role: RoleName, provider_name: str) -> None:
        self._roles[role].provider_name = provider_name
        self.query_one(f"#{role}-include-all", Checkbox).disabled = False
        if self._confirm_before_discovery and not self._discovery_confirmed:
            self._pending_provider_changes[role] = provider_name
            if not self._awaiting_confirmation:
                self._open_discovery_confirmation(role, provider_name)
            return
        self._start_model_load(role, provider_name)

    def _open_discovery_confirmation(self, role: RoleName, provider_name: str) -> None:
        try:
            candidate = build_picker_provider(provider_name)
        except ValueError as exc:
            self._set_role_error(role, _one_line(exc))
            return
        providers = dict(self._initial_providers)
        providers[role] = candidate
        self._awaiting_confirmation = True
        self.app.push_screen(
            ProviderConfirmationScreen(
                leader_provider=providers["leader"],
                subagent_provider=providers["subagent"],
                before_discovery=True,
            ),
            self._discovery_confirmation_finished,
        )

    def _discovery_confirmation_finished(self, accepted: bool | None) -> None:
        self._awaiting_confirmation = False
        if not accepted:
            self.dismiss(None)
            return
        self._discovery_confirmed = True
        if self._on_real_providers_confirmed is not None:
            self._on_real_providers_confirmed()
        pending = list(self._pending_provider_changes.items())
        self._pending_provider_changes.clear()
        for role, provider_name in pending:
            selected = self.query_one(f"#{role}-provider", Select).value
            if selected == provider_name:
                self._start_model_load(role, provider_name)

    def _start_model_load(self, role: RoleName, provider_name: str) -> None:
        state = self._roles[role]
        try:
            provider = build_picker_provider(provider_name)
        except ValueError as exc:
            self._set_role_error(role, _one_line(exc))
            return

        state.provider_name = provider_name
        state.default_model = _default_model(provider)
        initial_provider = self._initial_providers[role]
        initial_model = (
            _default_model(initial_provider)
            if initial_provider is not None and initial_provider.name == provider_name
            else ""
        )
        state.selected_model = initial_model or state.default_model
        state.loading = True
        state.models = []
        state.include_all = self.query_one(f"#{role}-include-all", Checkbox).value
        state.load_generation += 1
        generation = state.load_generation

        model_select = self.query_one(f"#{role}-model", Select)
        model_select.set_options([])
        model_select.disabled = True
        model_input = self.query_one(f"#{role}-manual-model", Input)
        model_input.disabled = False
        model_input.value = state.selected_model
        filter_text = "all models" if state.include_all else "text/coding models"
        self._set_role_status(role, f"Loading {provider.name} {filter_text}...")
        self._set_picker_error("")
        self._update_start_button()
        self._load_models(role, provider_name, state.include_all, generation)

    @work(thread=True, group="model-discovery")
    def _load_models(
        self,
        role: RoleName,
        provider_name: str,
        include_all: bool,
        generation: int,
    ) -> None:
        # list_models() may perform blocking HTTP. Keep it off Textual's event
        # loop and marshal results back to the UI thread.
        try:
            provider = build_picker_provider(provider_name)
            models = list_models(provider, include_all=include_all)
        except ProviderError as exc:
            self._call_ui_from_worker(
                self._finish_model_load,
                role,
                provider_name,
                include_all,
                generation,
                [],
                _one_line(exc),
            )
        except Exception as exc:  # noqa: BLE001
            self._call_ui_from_worker(
                self._finish_model_load,
                role,
                provider_name,
                include_all,
                generation,
                [],
                f"Unexpected {type(exc).__name__}: {_one_line(exc)}",
            )
        else:
            reason = "" if models else "provider returned no models"
            self._call_ui_from_worker(
                self._finish_model_load,
                role,
                provider_name,
                include_all,
                generation,
                models,
                reason,
            )

    def _call_ui_from_worker(self, callback, *args) -> None:  # noqa: ANN001
        if not self.app.is_running:
            return
        try:
            self.app.call_from_thread(callback, *args)
        except RuntimeError:
            return

    def _finish_model_load(
        self,
        role: RoleName,
        provider_name: str,
        include_all: bool,
        generation: int,
        models: list[str],
        reason: str,
    ) -> None:
        state = self._roles[role]
        if (
            state.provider_name != provider_name
            or state.include_all != include_all
            or state.load_generation != generation
        ):
            return

        state.loading = False
        state.models = list(models)
        model_select = self.query_one(f"#{role}-model", Select)
        if models:
            model_select.set_options([(model, model) for model in models])
            model_select.disabled = False
            filter_text = "all models" if include_all else "text/coding models"
            self._set_role_status(role, f"Showing {len(models)} {provider_name} {filter_text}.")
            if state.selected_model in models:
                model_select.value = state.selected_model
        else:
            model_select.set_options([])
            model_select.disabled = True
            self._set_role_status(
                role,
                f"Could not list {provider_name} models: {reason}. Enter a model manually.",
            )
        self.query_one(f"#{role}-manual-model", Input).disabled = False
        self._update_start_button()

    def _reset_role(self, role: RoleName) -> None:
        state = self._roles[role]
        state.provider_name = None
        state.default_model = ""
        state.selected_model = ""
        state.include_all = False
        state.loading = False
        state.models = []
        state.load_generation += 1
        self.query_one(f"#{role}-include-all", Checkbox).value = False
        self.query_one(f"#{role}-include-all", Checkbox).disabled = True
        model_select = self.query_one(f"#{role}-model", Select)
        model_select.set_options([])
        model_select.disabled = True
        model_input = self.query_one(f"#{role}-manual-model", Input)
        model_input.value = ""
        model_input.disabled = True
        self._set_role_status(role, f"Select a {role} provider.")
        self._update_start_button()

    def _set_role_error(self, role: RoleName, message: str) -> None:
        state = self._roles[role]
        state.loading = False
        state.selected_model = state.default_model
        self._set_role_status(role, message)
        self._update_start_button()

    def _set_role_status(self, role: RoleName, message: str) -> None:
        self.query_one(f"#{role}-model-status", Static).update(message)

    def _set_picker_error(self, message: str) -> None:
        self.query_one("#picker-error", Static).update(message)

    def _update_start_button(self) -> None:
        ready = all(self._role_ready(role) for role in self._roles)
        self.query_one("#start-chat", Button).disabled = not ready

    def _role_ready(self, role: RoleName) -> bool:
        state = self._roles[role]
        return bool(state.provider_name and state.selected_model and not state.loading)

    def _build_result(self) -> ProviderPickerResult | None:
        try:
            leader_provider = self._build_role_provider("leader")
            subagent_provider = self._build_role_provider("subagent")
        except ValueError as exc:
            self._set_picker_error(_one_line(exc))
            return None
        return ProviderPickerResult(
            leader_provider=leader_provider,
            subagent_provider=subagent_provider,
        )

    def _build_role_provider(self, role: RoleName) -> ModelProvider:
        state = self._roles[role]
        if not state.provider_name:
            raise ValueError(f"Choose a {role} provider.")
        if not state.selected_model:
            raise ValueError(f"Enter a {role} model.")
        return build_picker_provider(state.provider_name, model=state.selected_model)

    def _role_from_id(self, widget_id: str, suffix: str) -> RoleName | None:
        if not widget_id.endswith(suffix):
            return None
        role = widget_id[: -len(suffix)]
        if role in self._roles:
            return role  # type: ignore[return-value]
        return None


class ProviderConfirmationScreen(Screen[bool]):
    """Confirm before real provider calls can be made."""

    BINDINGS = [("escape", "cancel", "Cancel")]

    def __init__(
        self,
        *,
        leader_provider: ModelProvider | None,
        subagent_provider: ModelProvider | None,
        before_discovery: bool = False,
    ) -> None:
        super().__init__()
        self._leader_provider = leader_provider
        self._subagent_provider = subagent_provider
        self._before_discovery = before_discovery

    def compose(self) -> ComposeResult:
        yield Static("Real provider confirmation", id="confirm-title")
        yield Static(
            "WARNING: real providers can make network calls and consume API quota/credits."
            + (
                " Model discovery will make a provider request after you continue."
                if self._before_discovery
                else ""
            ),
            id="confirm-warning",
        )
        leader_summary = (
            provider_summary(self._leader_provider) if self._leader_provider else "Not selected"
        )
        subagent_summary = (
            provider_summary(self._subagent_provider) if self._subagent_provider else "Not selected"
        )
        yield Static(f"Leader: {leader_summary}", id="confirm-leader")
        yield Static(f"Subagents: {subagent_summary}", id="confirm-subagent")
        yield Button("Continue", id="confirm-continue", variant="warning")
        yield Button("Cancel", id="confirm-cancel")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#confirm-cancel", Button).focus()

    def action_cancel(self) -> None:
        self.dismiss(False)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "confirm-continue":
            self.dismiss(True)
        elif event.button.id == "confirm-cancel":
            self.dismiss(False)
