"""
Pre-configured run settings loader.

Reads ``run_config.json`` from the project root.  When ``use_config_file`` is
true, every interactive menu prompt is short-circuited and the value from the
file is used instead.  Set ``use_config_file`` to ``false`` to restore the
classic interactive menu flow.

Field format
────────────
Each field in the JSON may be either a plain scalar (legacy) or a dict with
the following keys:

  {
    "value":       <the actual value>,
    "options":     [<allowed_value>, ...],   # optional
    "description": "<human-readable explanation>"  # optional
  }

If ``options`` is present and the stored ``value`` is not in the list, the
loader prints a warning (with the description) and falls back to the default,
so misconfigured values are always caught at startup.

Any key may be omitted/left empty — the menu will fall back to its interactive
prompt for that field.  Comment-style ``//key`` entries (legacy) are ignored.
"""

import json
import os
from typing import Any, Callable

_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "run_config.json",
)

_MISSING = object()  # sentinel for "key not found in data"


class _RunConfig:
    def __init__(self):
        self._data: dict = {}
        self._loaded = False
        self._banner_shown = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        if not os.path.exists(_CONFIG_FILE):
            return
        try:
            with open(_CONFIG_FILE, "r", encoding="utf-8") as f:
                self._data = json.load(f)
        except Exception as e:
            print(f"[run_config] Warning: could not load {_CONFIG_FILE}: {e}")
            return
        self._apply_active_modes()

    def _apply_active_modes(self) -> None:
        """
        For every top-level section that has ``active_mode`` + ``modes``,
        merge the selected mode's values into the flat section namespace and
        set ``run_mode`` to the active mode key (the key IS the run type).

        Example in run_config.json::

            "global_correlation": {
                "active_mode": "pilot",
                "modes": {
                    "pilot":  {"scoring_provider": "inference_api", "queries_to_load": 3, ...},
                    "full":   {"scoring_provider": "gemini",    "queries_to_load": 200, ...},
                    "staged": {...},
                    "sync":   {...},
                    "analyze": {...}
                }
            }

        After merging, ``global_correlation.run_mode`` resolves to ``"pilot"``
        and all mode-specific fields are available as flat keys.
        Mode keys: pilot | full | staged | sync | analyze
        """
        for section_name, section in list(self._data.items()):
            if not isinstance(section, dict):
                continue
            active_mode = section.get("active_mode")
            modes = section.get("modes")
            if not active_mode or not isinstance(modes, dict):
                continue
            mode_values = modes.get(str(active_mode))
            if not isinstance(mode_values, dict):
                print(
                    f"[run_config] Warning: active_mode={active_mode!r} not found "
                    f"in {section_name}.modes — ignoring."
                )
                continue
            # The mode key itself is the run_mode value (pilot/full/staged/sync).
            section.setdefault("run_mode", str(active_mode))
            for k, v in mode_values.items():
                section[k] = v
            print(f"[run_config] {section_name}: active_mode={active_mode!r} applied.")

    def _show_banner(self) -> None:
        if self._banner_shown:
            return
        self._banner_shown = True
        print("\n[CONFIG] Using pre-configured values from run_config.json")
        print(f"[CONFIG] File: {_CONFIG_FILE}")
        print("[CONFIG] Set use_config_file=false to switch to interactive menus.\n")

    # ------------------------------------------------------------------
    # Internal field access
    # ------------------------------------------------------------------

    def _get_field_raw(self, dotted_key: str):
        """
        Navigate to a field by dotted path and return the raw stored value.
        Returns the ``_MISSING`` sentinel when the key is absent.
        """
        self._load()
        parts = dotted_key.split(".")
        val = self._data
        for p in parts:
            if not isinstance(val, dict) or p not in val:
                return _MISSING
            val = val[p]
        return val

    def _unwrap(self, key: str, raw: Any, default: Any) -> Any:
        """
        Unwrap a dict-format field, validate against ``options``, and return
        the resolved value.  Falls back to ``default`` when:
          - the value is empty / None
          - ``options`` is specified and the value is not in the list
            (a warning is printed in that case)

        Plain scalars (legacy format) are returned as-is.
        """
        if isinstance(raw, dict) and "value" in raw:
            field_val = raw["value"]
            options = raw.get("options")
            description = raw.get("description", "")

            if field_val == "" or field_val is None:
                return default

            if options is not None and field_val not in options:
                print(
                    f"[config] ⚠  {key} = {field_val!r} is not in allowed options {options}."
                )
                if description:
                    short = description[:120] + ("…" if len(description) > 120 else "")
                    print(f"[config]    {short}")
                return default

            return field_val

        # Plain scalar (not wrapped)
        if raw == "" or raw is None:
            return default
        return raw

    def _field_description(self, dotted_key: str) -> str:
        """Return the ``description`` string for a field, or '' if absent."""
        raw = self._get_field_raw(dotted_key)
        if raw is _MISSING or not isinstance(raw, dict):
            return ""
        return raw.get("description", "")

    def _print_config_value(self, key: str, val: Any) -> None:
        """Print a resolved config value, showing the banner (once) and a short description."""
        self._show_banner()
        desc = self._field_description(key)
        if desc:
            short = desc[:80] + ("…" if len(desc) > 80 else "")
            print(f"[config] {key} = {val!r}  # {short}")
        else:
            print(f"[config] {key} = {val!r}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        self._load()
        return bool(self.get("use_config_file", False))

    def get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Retrieve a value by dot-separated path (e.g. ``global_correlation.retrieval_top_k``).

        Supports both the new dict format ``{"value": ..., "options": [...], "description": "..."}``
        and plain scalar values (legacy).  Returns ``default`` when the key is
        absent, empty, or fails options validation.
        """
        raw = self._get_field_raw(dotted_key)
        if raw is _MISSING:
            return default
        return self._unwrap(dotted_key, raw, default)

    def get_field(self, path: list[str], default: Any = None) -> Any:
        """
        Retrieve a field using a list of path components instead of a dotted string.

        Example::

            run_config.get_field(["global_correlation", "retrieval_top_k"], default=100)
        """
        return self.get(".".join(path), default)

    def has(self, dotted_key: str) -> bool:
        """True when the key is explicitly set to a non-empty, valid value."""
        return self.get(dotted_key, None) is not None

    def get_str(self, key: str, default: str | None = None) -> str | None:
        val = self.get(key, default)
        return str(val) if val is not None else default

    def get_int(self, key: str, default: int = 0) -> int:
        val = self.get(key, default)
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    def get_float(self, key: str, default: float = 0.0) -> float:
        val = self.get(key, default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool | None = None) -> bool | None:
        val = self.get(key)
        if val is None:
            return default
        if isinstance(val, bool):
            return val
        return str(val).lower() in ("true", "1", "yes", "y")

    # ------------------------------------------------------------------
    # Menu helpers — resolve from config or fall back to interactive prompt
    # ------------------------------------------------------------------

    def menu_str(self, key: str, prompt_fn: Callable[[], str], default: str | None = None) -> str:
        """
        Resolve a string value from config or via the supplied prompt callable.
        Prints the resolved value with its description when sourced from config.
        """
        if self.enabled:
            val = self.get_str(key)
            if val is not None:
                self._print_config_value(key, val)
                return val
        return prompt_fn()

    def menu_bool(self, key: str, prompt_fn: Callable[[], bool]) -> bool:
        """Bool variant of :meth:`menu_str`."""
        if self.enabled:
            val = self.get_bool(key)
            if val is not None:
                self._print_config_value(key, val)
                return val
        return prompt_fn()

    def menu_int(self, key: str, prompt_fn: Callable[[], int], default: int | None = None) -> int:
        """Int variant of :meth:`menu_str`."""
        if self.enabled and self.has(key):
            val = self.get_int(key, default if default is not None else 0)
            self._print_config_value(key, val)
            return val
        return prompt_fn()


run_config = _RunConfig()
