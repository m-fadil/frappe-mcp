"""Host integration: transactions, error reporting and JSON encoding.

Frappe is optional. Inside a Frappe request these helpers use the request's
database connection, Error Log and JSON conventions; elsewhere they fall back
to plain Python behaviour.
"""

from __future__ import annotations

import datetime
import decimal
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

try:
    import frappe
except ImportError:  # pragma: no cover - exercised only without Frappe
    frappe = None

logger = logging.getLogger('frappe_mcp')

_SAVEPOINT = 'frappe_mcp_request'
# The connection whose savepoint the current request opened, if any. A
# ContextVar, so concurrent request threads never see each other's state.
_scope_db: ContextVar[Any] = ContextVar('frappe_mcp_scope_db', default=None)


class ToolError(Exception):
    """Raise from a tool to return its message to the client as `isError: true`.

    Other exceptions are logged and reported to the client with a generic
    message so that internals (SQL, paths, stack details) do not leak.
    """


def _db():
    if frappe is None:
        return None
    return getattr(frappe.local, 'db', None)


@contextmanager
def request_scope() -> Iterator[None]:
    """Run one MCP request's handler inside a database savepoint.

    Only inside this scope does `rollback()` act. Handlers called directly
    (e.g. from tests) leave the transaction to their caller.
    """
    db = _db()
    if db is None:
        yield
        return
    db.savepoint(_SAVEPOINT)
    token = _scope_db.set(db)
    try:
        yield
    finally:
        _scope_db.reset(token)


def rollback() -> None:
    """Undo the database work of the current request scope.

    Frappe commits a successful POST request. MCP reports failures inside a
    successful HTTP response, so without this a failing tool's partial writes
    would be committed.
    """
    db = _scope_db.get()
    if db is None:
        return
    try:
        db.rollback(save_point=_SAVEPOINT)
    except Exception:
        # The handler committed, which ends the savepoint: discard the
        # uncommitted work that followed. Committed work cannot be undone.
        db.rollback()


def user_message(error: BaseException) -> str | None:
    """The message of an error meant for the caller, or None if it is internal."""
    if isinstance(error, ToolError):
        return str(error)
    if frappe is not None and isinstance(
        error, frappe.ValidationError | frappe.PermissionError
    ):
        return str(error) or type(error).__name__
    return None


def log_exception(title: str) -> None:
    """Record the exception being handled where the host keeps errors."""
    if _db() is not None:
        try:
            frappe.log_error(title=title)
            return
        except Exception:
            pass
    logger.exception(title)


def json_default(obj: Any) -> Any:
    """`json.dumps` default: Frappe's encoder when available, else common types."""
    if frappe is not None:
        from frappe.utils.response import json_handler

        try:
            return json_handler(obj)
        except TypeError:
            pass
    if isinstance(obj, datetime.date | datetime.time):
        return obj.isoformat()
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, set | frozenset):
        return list(obj)
    return str(obj)  # UUID, Path and anything else with a meaningful str()
