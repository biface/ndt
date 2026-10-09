"""
Compatibility helpers for Python versions older than the target baseline.

Private module. Each helper here backports a standard library feature that
the minimum supported Python version lacks, without adding a runtime
dependency. A helper is removed as soon as the minimum version provides the
feature natively.

``override``
    PEP 698 decorator, in ``typing`` from Python 3.12. Type checkers resolve
    it from ``typing_extensions`` (bundled typeshed stubs); at runtime the
    standard version is used on 3.12+ and ``_override`` on 3.11. To be
    removed in 1.4.0 (#79), when the minimum becomes 3.12.
"""

import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

_F = TypeVar("_F", bound=Callable[..., object])


def _override(method: _F, /) -> _F:
    """
    Mark a method as overriding a base class method (PEP 698 fallback).

    Sets ``__override__ = True`` on the decorated object when possible and
    returns it unchanged, as ``typing.override`` does. Objects that do not
    accept attributes are returned without the marker.

    Parameters
    ----------
    method : Callable
        The overriding method.

    Returns
    -------
    Callable
        The same object.
    """
    try:
        method.__override__ = True  # pyright: ignore[reportFunctionMemberAccess]
    except (AttributeError, TypeError):
        pass
    return method


if TYPE_CHECKING:
    from typing_extensions import override as override
elif sys.version_info >= (3, 12):
    from typing import override as override
else:
    override = _override

__all__ = ["override"]
