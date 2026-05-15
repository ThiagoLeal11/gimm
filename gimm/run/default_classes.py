import functools
from typing import Any, Dict
from dataclasses import fields


class TrackOverridesMixin:
    """Mixin to expose tracking methods and attributes to static type checkers."""
    _user_overrides: Dict[str, Any]

    def get_user_overrides(self) -> Dict[str, Any]:
        return getattr(self, "_user_overrides", {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            field.name: getattr(self, field.name)
            for field in fields(type(self))
        }

    def set(self, values: Dict[str, Any]) -> None:
        for field in fields(type(self)):
            if field.name in values:
                setattr(self, field.name, values[field.name])


def track_overrides(cls):
    """Decorator to track explicitly provided __init__ arguments."""
    original_init = cls.__init__

    @functools.wraps(original_init)
    def new_init(self, *args, **kwargs):
        cls_fields = fields(cls)

        explicit_inputs = {cls_fields[i].name: arg for i, arg in enumerate(args)}
        explicit_inputs.update(kwargs)

        self._user_overrides = explicit_inputs
        original_init(self, *args, **kwargs)

    cls.__init__ = new_init
    return cls