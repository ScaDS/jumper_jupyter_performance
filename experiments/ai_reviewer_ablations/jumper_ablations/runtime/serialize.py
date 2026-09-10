"""Making the reviewer's payloads safe to write as JSON.

The collected context holds whatever the metric collectors produced - numpy
scalars, arrays, pandas objects - and a record has to survive a round trip
through a file. Values that cannot be represented are turned into their repr
rather than dropped: a metric reading a record should be able to tell "this
source was empty" from "this source held something we could not store".
"""
from __future__ import annotations

import math
from typing import Any


def jsonable(value: Any) -> Any:
    """A JSON-representable copy of *value*."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]

    for attribute in ("item", "tolist"):
        converter = getattr(value, attribute, None)
        if callable(converter):
            try:
                return jsonable(converter())
            except Exception:
                pass

    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return jsonable(to_dict())
        except Exception:
            pass

    return repr(value)


def messages_as_dicts(messages: list) -> list[dict]:
    """LangChain messages as ``{role, content}``, verbatim.

    Verbatim matters: these are what a judge is shown as "the sources the
    model had", and a reformatted copy would quietly weaken that claim.
    """
    serialised = []
    for message in messages or []:
        content = getattr(message, "content", "")
        serialised.append(
            {
                "role": type(message).__name__,
                "content": content if isinstance(content, str) else jsonable(content),
            }
        )
    return serialised
