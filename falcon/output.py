"""Stable, ANSI-free serializers for Falcon's noninteractive commands."""

from __future__ import annotations

import dataclasses
import json
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = "falcon/v1"


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: jsonable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, (Path, Enum)):
        return str(value.value if isinstance(value, Enum) else value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def envelope(
    kind: str,
    data: Any,
    *,
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "data": jsonable(data),
    }
    if meta:
        result["meta"] = jsonable(meta)
    return result


def dumps(
    kind: str,
    data: Any,
    *,
    meta: Optional[Mapping[str, Any]] = None,
    pretty: bool = False,
) -> str:
    """Serialize exactly one JSON object with deterministic keys."""
    return json.dumps(
        envelope(kind, data, meta=meta),
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
        sort_keys=True,
    )


def truncate(value: Any, width: int) -> str:
    text = "-" if value is None else str(value)
    if width <= 0:
        return ""
    return text if len(text) <= width else text[: max(1, width - 1)] + "…"


def render_table(
    headers: Sequence[str],
    rows: Iterable[Sequence[Any]],
    *,
    maximum_widths: Optional[Sequence[int]] = None,
) -> str:
    """Render a compact plain-text table suitable for terminals and logs."""
    materialized = [[str("-" if cell is None else cell) for cell in row] for row in rows]
    count = len(headers)
    widths = [len(str(header)) for header in headers]
    for row in materialized:
        if len(row) != count:
            raise ValueError("table row has a different number of cells than headers")
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    if maximum_widths:
        if len(maximum_widths) != count:
            raise ValueError("maximum_widths must match headers")
        widths = [
            min(width, max(1, int(maximum)))
            for width, maximum in zip(widths, maximum_widths)
        ]
    lines = [
        "  ".join(
            truncate(header, widths[index]).ljust(widths[index])
            for index, header in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    lines.extend(
        "  ".join(
            truncate(value, widths[index]).ljust(widths[index])
            for index, value in enumerate(row)
        ).rstrip()
        for row in materialized
    )
    return "\n".join(lines)
