from __future__ import annotations

import copy
from typing import Any, Callable

from PyQt6 import QtCore, QtGui

from anylabeling.config import save_config

from .schema import (
    SETTING_FIELD_MAP,
    SETTING_FIELDS,
    SHORTCUT_DUPLICATE_WHITELIST,
    get_nested_value,
    set_nested_value,
    defaults_map,
    fields_for_page,
)


class SettingsValidationError(ValueError):
    def __init__(self, message: str, conflict_keys: list[str] | None = None):
        super().__init__(message)
        self.conflict_keys = conflict_keys or []


class SettingsController(QtCore.QObject):
    field_changed = QtCore.pyqtSignal(str, object)
    save_succeeded = QtCore.pyqtSignal()
    save_failed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        config: dict[str, Any],
        apply_callback: Callable[[str, Any], None] | None = None,
        save_callback: Callable[[dict[str, Any]], Any] | None = None,
        parent: QtCore.QObject | None = None,
        save_delay_ms: int = 300,
        defer_runtime_apply: bool = False,
    ):
        super().__init__(parent)
        self._config = config
        self._defer_runtime_apply = bool(defer_runtime_apply)
        self._working_config = (
            copy.deepcopy(config) if self._defer_runtime_apply else config
        )
        self._apply_callback = apply_callback
        self._save_callback = save_callback or save_config
        self._field_map = SETTING_FIELD_MAP
        self._defaults = defaults_map()
        self._dirty_keys: set[str] = set()
        self._last_saved_keys: list[str] = []
        self._save_timer = QtCore.QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(save_delay_ms)
        self._save_timer.timeout.connect(self._save_now)

    @property
    def fields(self):
        return SETTING_FIELDS

    @property
    def keys(self) -> list[str]:
        return list(self._field_map.keys())

    def get_value(self, key: str) -> Any:
        return get_nested_value(self._working_config, key)

    @property
    def last_saved_keys(self) -> list[str]:
        return list(self._last_saved_keys)

    def get_page_keys(self, primary: str, secondary: str) -> list[str]:
        return [field.key for field in fields_for_page(primary, secondary)]

    def update_field(
        self,
        key: str,
        value: Any,
        schedule_save: bool = True,
        emit_signal: bool = True,
    ) -> bool:
        if key not in self._field_map:
            raise KeyError(key)
        field = self._field_map[key]
        normalized = self._validate_and_normalize(field, value)
        old_value = copy.deepcopy(self.get_value(key))
        if old_value == normalized:
            return False

        set_nested_value(self._working_config, key, copy.deepcopy(normalized))
        if self._defer_runtime_apply:
            runtime_value = get_nested_value(self._config, key)
            if runtime_value == normalized:
                self._dirty_keys.discard(key)
            else:
                self._dirty_keys.add(key)
        else:
            try:
                if self._apply_callback is not None:
                    self._apply_callback(key, copy.deepcopy(normalized))
            except Exception:
                set_nested_value(self._working_config, key, old_value)
                raise

        if emit_signal:
            self.field_changed.emit(key, copy.deepcopy(normalized))
        if schedule_save:
            self._save_timer.start()
        return True

    def reset_page(self, primary: str, secondary: str) -> None:
        self.reset_keys(self.get_page_keys(primary, secondary))

    def reset_all(self) -> None:
        self.reset_keys(self.keys)

    def reset_keys(self, keys: list[str]) -> None:
        changed = False
        first_error: Exception | None = None
        for key in keys:
            if key not in self._defaults:
                continue
            default_value = copy.deepcopy(self._defaults[key])
            try:
                changed = (
                    self.update_field(
                        key,
                        default_value,
                        schedule_save=False,
                        emit_signal=True,
                    )
                    or changed
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        if changed:
            self._save_timer.start()
        if first_error is not None:
            raise first_error

    def has_pending_save(self) -> bool:
        return self._save_timer.isActive()

    def has_unsaved_changes(self) -> bool:
        return bool(self._dirty_keys) if self._defer_runtime_apply else False

    def save_now(self) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
        self._save_now()

    def get_default_value(self, key: str) -> Any:
        if key not in self._defaults:
            raise KeyError(key)
        return copy.deepcopy(self._defaults[key])

    def flush(self) -> None:
        if self._save_timer.isActive():
            self._save_timer.stop()
            self._save_now()

    def discard_changes(self) -> None:
        if not self._defer_runtime_apply:
            return
        self._working_config = copy.deepcopy(self._config)
        self._dirty_keys.clear()
        self._last_saved_keys = []

    def close_session(self) -> None:
        if self._defer_runtime_apply:
            self.discard_changes()
            return
        self.flush()

    def _save_now(self) -> None:
        pending_config = (
            copy.deepcopy(self._working_config)
            if self._defer_runtime_apply
            else self._config
        )
        changed_keys = (
            sorted(self._dirty_keys) if self._defer_runtime_apply else []
        )
        try:
            result = self._save_callback(pending_config)
            if result is False:
                raise RuntimeError("save callback returned False")
            if self._defer_runtime_apply:
                self._config.clear()
                self._config.update(copy.deepcopy(pending_config))
                self._working_config = copy.deepcopy(self._config)
                if self._apply_callback is not None:
                    for key in changed_keys:
                        self._apply_callback(
                            key,
                            copy.deepcopy(get_nested_value(self._config, key)),
                        )
        except Exception as exc:
            self.save_failed.emit(str(exc))
            return
        if self._defer_runtime_apply:
            self._dirty_keys.clear()
            self._last_saved_keys = changed_keys
        else:
            self._last_saved_keys = []
        self.save_succeeded.emit()

    def _validate_and_normalize(self, field, value: Any) -> Any:
        control = field.control
        if control == "bool":
            if not isinstance(value, bool):
                raise SettingsValidationError(
                    f"{field.label} requires a boolean value"
                )
            return value

        if control == "enum":
            if value is None and field.allow_none:
                normalized = None
            elif value in field.options:
                normalized = value
            else:
                raise SettingsValidationError(
                    f"{field.label} has unsupported option: {value}"
                )
            if field.key.startswith("shortcuts."):
                self._validate_shortcut_conflicts(field.key, normalized)
            return normalized

        if control == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise SettingsValidationError(
                    f"{field.label} requires an integer value"
                )
            if field.minimum is not None and value < field.minimum:
                raise SettingsValidationError(
                    f"{field.label} must be >= {field.minimum}"
                )
            if field.maximum is not None and value > field.maximum:
                raise SettingsValidationError(
                    f"{field.label} must be <= {field.maximum}"
                )
            return value

        if control == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SettingsValidationError(
                    f"{field.label} requires a numeric value"
                )
            numeric = float(value)
            if field.minimum is not None and numeric < field.minimum:
                raise SettingsValidationError(
                    f"{field.label} must be >= {field.minimum}"
                )
            if field.maximum is not None and numeric > field.maximum:
                raise SettingsValidationError(
                    f"{field.label} must be <= {field.maximum}"
                )
            return numeric

        if control == "str":
            if value in (None, "") and field.allow_none:
                return None
            if not isinstance(value, str):
                raise SettingsValidationError(
                    f"{field.label} requires a string value"
                )
            normalized = value.strip()
            if field.key == "canvas.crosshair.color":
                if not normalized:
                    raise SettingsValidationError(
                        f"{field.label} requires a HEX color value"
                    )
                if not normalized.startswith("#"):
                    normalized = f"#{normalized}"
                color = QtGui.QColor(normalized)
                if not color.isValid():
                    raise SettingsValidationError(
                        f"{field.label} requires a valid HEX color"
                    )
                return color.name(QtGui.QColor.NameFormat.HexRgb).upper()
            return normalized

        if control == "color":
            if not isinstance(value, (list, tuple)):
                raise SettingsValidationError(
                    f"{field.label} requires a color list"
                )
            channels = list(value)
            if len(channels) != field.channels:
                raise SettingsValidationError(
                    f"{field.label} requires {field.channels} channels"
                )
            normalized = []
            for channel in channels:
                if isinstance(channel, bool) or not isinstance(channel, int):
                    raise SettingsValidationError(
                        f"{field.label} channels must be integers"
                    )
                if channel < 0 or channel > 255:
                    raise SettingsValidationError(
                        f"{field.label} channels must be in [0, 255]"
                    )
                normalized.append(channel)
            return normalized

        if control == "vector2":
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise SettingsValidationError(
                    f"{field.label} requires two numeric values"
                )
            result: list[float] = []
            for item in value:
                if isinstance(item, bool) or not isinstance(
                    item, (int, float)
                ):
                    raise SettingsValidationError(
                        f"{field.label} requires two numeric values"
                    )
                result.append(float(item))
            return result

        if control == "shortcut":
            normalized = self._normalize_shortcut_value(value)
            if normalized is None:
                return None
            self._validate_shortcut_conflicts(field.key, normalized)
            return normalized

        if control == "multi_shortcut":
            normalized = self._normalize_multi_shortcut_value(value)
            if not normalized:
                return []
            self._validate_shortcut_conflicts(field.key, normalized)
            return normalized

        raise SettingsValidationError(
            f"Unsupported control type for {field.key}: {control}"
        )

    def _normalize_shortcut_value(self, value: Any) -> str | None:
        if value in (None, ""):
            return None
        if isinstance(value, (list, tuple)):
            values = [item for item in value if item not in (None, "")]
            if not values:
                return None
            value = values[0]
        sequence = QtGui.QKeySequence(str(value)).toString(
            QtGui.QKeySequence.SequenceFormat.PortableText
        )
        sequence = sequence.strip()
        if not sequence:
            raise SettingsValidationError("Invalid shortcut sequence")
        return sequence

    def _normalize_multi_shortcut_value(self, value: Any) -> list[str]:
        if value in (None, ""):
            return []
        if isinstance(value, (list, tuple)):
            raw_values = [item for item in value if item not in (None, "")]
        else:
            raw_values = [value]
        normalized: list[str] = []
        for raw in raw_values:
            sequence = self._normalize_shortcut_value(raw)
            if sequence and sequence not in normalized:
                normalized.append(sequence)
        return normalized

    def _shortcut_sequences_for(
        self,
        key: str,
        override_key: str,
        override_value: Any,
    ) -> list[str]:
        field = self._field_map[key]
        value = override_value if key == override_key else self.get_value(key)
        if field.control == "multi_shortcut":
            return self._normalize_multi_shortcut_value(value)
        normalized = self._normalize_shortcut_value(value)
        return [normalized] if normalized else []

    def _validate_shortcut_conflicts(
        self,
        override_key: str,
        override_value: Any,
    ) -> None:
        shortcut_keys = [
            field.key
            for field in SETTING_FIELDS
            if field.primary == "Shortcuts"
        ]
        sequence_to_keys: dict[str, list[str]] = {}
        for key in shortcut_keys:
            for sequence in self._shortcut_sequences_for(
                key,
                override_key,
                override_value,
            ):
                sequence_to_keys.setdefault(sequence, []).append(key)

        for sequence, assigned_keys in sequence_to_keys.items():
            unique_keys = sorted(set(assigned_keys))
            if len(unique_keys) <= 1:
                continue
            if frozenset(unique_keys) in SHORTCUT_DUPLICATE_WHITELIST:
                continue
            source_key = (
                override_key if override_key in unique_keys else unique_keys[0]
            )
            conflict_keys = [key for key in unique_keys if key != source_key]
            if not conflict_keys:
                conflict_keys = unique_keys[1:]
            labels = [
                (
                    f"'{self._field_map[key].label}'"
                    f" ({self._field_map[key].secondary})"
                )
                for key in conflict_keys
            ]
            conflict_message = (
                f"Shortcut '{sequence}' conflicts with " + ", ".join(labels)
            )
            raise SettingsValidationError(
                conflict_message,
                conflict_keys=unique_keys,
            )
