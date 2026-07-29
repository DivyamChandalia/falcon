"""Kubernetes quantity parsing and stable Falcon resource formatting."""

from __future__ import annotations

import re
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, InvalidOperation
from typing import Callable, Optional, Tuple


class QuantityError(ValueError):
    """Raised when a Kubernetes resource quantity is malformed."""


_QUANTITY = re.compile(
    r"^(?P<number>[+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))"
    r"(?P<suffix>Ki|Mi|Gi|Ti|Pi|Ei|[eE][+-]?[0-9]+|n|u|m|[kKMGTEP])?$"
)
_DECIMAL_FACTORS = {
    "n": Decimal("1e-9"),
    "u": Decimal("1e-6"),
    "m": Decimal("1e-3"),
    "": Decimal(1),
    "k": Decimal(1000),
    "K": Decimal(1000),
    "M": Decimal(1000) ** 2,
    "G": Decimal(1000) ** 3,
    "T": Decimal(1000) ** 4,
    "P": Decimal(1000) ** 5,
    "E": Decimal(1000) ** 6,
}
_BINARY_POWERS = {"Ki": 1, "Mi": 2, "Gi": 3, "Ti": 4, "Pi": 5, "Ei": 6}
_MIB = Decimal(1024) ** 2


def parse_quantity(value: str) -> Decimal:
    """Parse a non-negative Kubernetes quantity into base units.

    Both binary SI (``Gi``) and decimal SI (``G``, ``m``) suffixes are
    supported, as are decimal exponents such as ``12e3``. The return type is
    :class:`~decimal.Decimal` so large byte quantities remain exact.
    """

    if not isinstance(value, str):
        raise QuantityError("quantity must be a string")
    raw = value.strip()
    match = _QUANTITY.fullmatch(raw)
    if not match:
        raise QuantityError(f"invalid Kubernetes quantity: {value!r}")
    try:
        amount = Decimal(match.group("number"))
    except InvalidOperation as exc:
        raise QuantityError(f"invalid Kubernetes quantity: {value!r}") from exc
    suffix = match.group("suffix") or ""
    if suffix in _BINARY_POWERS:
        result = amount * (Decimal(1024) ** _BINARY_POWERS[suffix])
    elif suffix.startswith(("e", "E")):
        result = amount * (Decimal(10) ** int(suffix[1:]))
    else:
        result = amount * _DECIMAL_FACTORS[suffix]
    if not result.is_finite() or result < 0:
        raise QuantityError(f"quantity must be a finite non-negative value: {value!r}")
    return result


def parse_cpu(value: str) -> Decimal:
    """Parse a CPU quantity into cores.

    Binary suffixes are rejected because they have no meaningful CPU
    interpretation. Falcon also rejects sub-millicore precision, matching the
    Kubernetes API's effective CPU resolution.
    """

    request = request_part(value)
    match = _QUANTITY.fullmatch(request.strip())
    if not match or (match.group("suffix") or "") in _BINARY_POWERS:
        raise QuantityError(f"invalid CPU quantity: {value!r}")
    result = parse_quantity(request)
    if result > 0 and result < Decimal("0.001"):
        raise QuantityError("CPU quantity must be at least 1m")
    if (result * 1000) != (result * 1000).to_integral_value():
        raise QuantityError("CPU quantity cannot use precision finer than 1m")
    return result


def parse_memory_bytes(value: str) -> Decimal:
    """Parse a memory quantity into bytes."""

    # Kubernetes accepts fractional binary quantities such as ``0.1Gi`` and
    # rounds them to byte precision when the API server canonicalizes the
    # manifest.  Keeping the Decimal here avoids rejecting a valid request
    # merely because its binary expansion is not an integer.
    return parse_quantity(request_part(value))


def parse_memory_gib(value: str) -> float:
    """Parse a memory quantity and return GiB for planning/display math."""

    return float(parse_memory_bytes(value) / (Decimal(1024) ** 3))


def request_part(value: str) -> str:
    """Return the request half of a compact ``request:limit`` value."""

    if not isinstance(value, str):
        raise QuantityError("resource value must be a string")
    request, separator, limit = value.partition(":")
    if not request.strip():
        raise QuantityError("resource request must not be empty")
    if separator and not limit.strip():
        raise QuantityError("resource limit must not be empty")
    if separator and ":" in limit:
        raise QuantityError(f"resource value has too many ':' separators: {value!r}")
    return request.strip()


def split_pair(
    value: str,
    parser: Callable[[str], Decimal],
    *,
    normalize_limit: bool = False,
) -> Tuple[str, Optional[str]]:
    """Parse and validate a compact resource request/limit pair."""

    request = request_part(value)
    _, separator, limit = value.partition(":")
    limit = limit.strip() if separator else None
    request_value = parser(request)
    if limit is not None:
        limit_value = parser(limit)
        if limit_value < request_value:
            raise QuantityError("resource limit must be greater than or equal to request")
    if normalize_limit:
        limit = request
    return request, limit


def format_cpu(value: Decimal | float | int) -> str:
    """Floor cores to 100m without ever formatting above available capacity."""

    amount = Decimal(str(value))
    if amount <= 0:
        raise QuantityError("CPU allocation must be positive")
    floored = (amount * 10).to_integral_value(rounding=ROUND_FLOOR) / 10
    floored = max(floored, Decimal("0.1"))
    return _decimal_text(floored)


def format_memory_gib(value: Decimal | float | int) -> str:
    """Floor GiB to an integral MiB quantity valid for Kubernetes memory."""

    amount = Decimal(str(value))
    if amount <= 0:
        raise QuantityError("memory allocation must be positive")
    mebibytes = (amount * 1024).to_integral_value(rounding=ROUND_FLOOR)
    return _format_integral_mebibytes(max(mebibytes, Decimal(1)))


def normalize_memory(value: str) -> str:
    """Return a byte-integral Kubernetes memory quantity.

    User-provided fractional binary quantities are rounded up to the nearest
    MiB so the normalized request is never smaller than requested. Values
    below one MiB retain byte precision.
    """

    amount = parse_memory_bytes(value)
    if amount <= 0:
        raise QuantityError("memory allocation must be positive")
    if amount >= _MIB:
        mebibytes = (amount / _MIB).to_integral_value(
            rounding=ROUND_CEILING
        )
        return _format_integral_mebibytes(mebibytes)
    return str(int(amount.to_integral_value(rounding=ROUND_CEILING)))


def _format_integral_mebibytes(mebibytes: Decimal) -> str:
    value = int(mebibytes)
    if value % 1024 == 0:
        return f"{value // 1024}Gi"
    return f"{value}Mi"


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered
