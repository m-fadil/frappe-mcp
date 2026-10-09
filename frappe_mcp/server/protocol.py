"""Protocol-era detection and Streamable HTTP request validation.

Implements the dual-era rules of MCP 2026-07-28:

- A *modern* request carries ``io.modelcontextprotocol/protocolVersion`` in
  ``params._meta`` and is served statelessly; its HTTP headers must mirror the
  body (``MCP-Protocol-Version``, ``Mcp-Method``, ``Mcp-Name``).
- A *legacy* request (no per-request version) keeps the ``initialize``-based
  behaviour of protocol versions 2025-11-25 and earlier; a non-legacy
  ``MCP-Protocol-Version`` header on it is rejected.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit

from frappe_mcp.server import types

META_PROTOCOL_VERSION = 'io.modelcontextprotocol/protocolVersion'
META_CLIENT_CAPABILITIES = 'io.modelcontextprotocol/clientCapabilities'
META_SERVER_INFO = 'io.modelcontextprotocol/serverInfo'

MODERN_VERSIONS = ('2026-07-28',)
LEGACY_VERSIONS = ('2025-11-25', '2025-06-18', '2025-03-26')
SUPPORTED_VERSIONS = MODERN_VERSIONS + LEGACY_VERSIONS
DEFAULT_LEGACY_VERSION = LEGACY_VERSIONS[0]

HEADER_PROTOCOL_VERSION = 'MCP-Protocol-Version'
HEADER_METHOD = 'Mcp-Method'
HEADER_NAME = 'Mcp-Name'

# Methods whose `Mcp-Name` header mirrors a body field.
NAME_FIELD_BY_METHOD = {
    'tools/call': 'name',
    'prompts/get': 'name',
    'resources/read': 'uri',
}

# Results that carry caching hints (`ttlMs`, `cacheScope`) in the modern era.
CACHEABLE_METHODS = frozenset({'server/discover', 'tools/list', 'prompts/list'})
# Always "private": the entry function may return a per-request tool registry
# (per user roles or site settings), so a catalogue must never be shared.
CACHE_SCOPE = 'private'
# One minute: the server sends no list_changed notifications, so the TTL is the
# only freshness signal. A short window spares repeated list calls within one
# conversation turn while settings changes still propagate quickly. Access is
# re-checked on every tools/call, so a stale catalogue cannot grant access.
DEFAULT_CACHE_TTL_MS = 60_000

_BASE64_PREFIX = '=?base64?'
_BASE64_SUFFIX = '?='
_DEFAULT_PORTS = {'http': 80, 'https': 443}


class ProtocolError(Exception):
    """A request rejected before dispatch, answered with HTTP 400."""

    def __init__(self, code: int, message: str, data: Any = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class HeaderMismatchError(ProtocolError):
    """Mirrored HTTP headers are missing, malformed or disagree with the body."""

    def __init__(self, message: str):
        super().__init__(types.HEADER_MISMATCH, message)


def resolve_era(headers: Mapping[str, str], method: str, params: Mapping) -> bool:
    """Validate the request's protocol metadata and return True if it is modern.

    `headers` must be case-insensitive (e.g. werkzeug's `Headers`).

    Raises:
        ProtocolError: if the requested version is unsupported, required
            `_meta` fields are missing or the HTTP headers do not match.
    """
    version = get_requested_version(params)
    if version is None:
        _check_legacy_header(headers, method)
        return False

    if version not in MODERN_VERSIONS:
        raise _unsupported_version(version)
    if not isinstance(params['_meta'].get(META_CLIENT_CAPABILITIES), dict):
        raise ProtocolError(
            types.INVALID_PARAMS,
            f'Invalid params: _meta is missing required {META_CLIENT_CAPABILITIES}',
        )
    validate_headers(headers, method, params)
    return True


def _check_legacy_header(headers: Mapping[str, str], method: str) -> None:
    # `initialize` precedes version negotiation, so its header is not binding.
    header = headers.get(HEADER_PROTOCOL_VERSION)
    if header is None or header in LEGACY_VERSIONS or method == 'initialize':
        return
    if header in MODERN_VERSIONS:
        raise ProtocolError(
            types.INVALID_PARAMS,
            f'Invalid params: _meta is missing required {META_PROTOCOL_VERSION}',
        )
    raise _unsupported_version(header)


def _unsupported_version(requested: Any) -> ProtocolError:
    return ProtocolError(
        types.UNSUPPORTED_PROTOCOL_VERSION,
        'Unsupported protocol version',
        data={'supported': list(SUPPORTED_VERSIONS), 'requested': requested},
    )


def get_requested_version(params: Any) -> Any:
    """Return the per-request protocol version, or None for a legacy request."""
    if not isinstance(params, Mapping):
        return None
    meta = params.get('_meta')
    if not isinstance(meta, Mapping):
        return None
    return meta.get(META_PROTOCOL_VERSION)


def validate_headers(headers: Mapping[str, str], method: str, params: Mapping) -> None:
    """Check the Streamable HTTP request metadata headers of a modern request.

    `headers` must be case-insensitive (e.g. werkzeug's `Headers`).

    Raises:
        HeaderMismatchError: if a required header is missing or does not match
            the corresponding body value.
    """
    version = params['_meta'][META_PROTOCOL_VERSION]
    _expect_header(headers, HEADER_PROTOCOL_VERSION, version)
    _expect_header(headers, HEADER_METHOD, method)

    field = NAME_FIELD_BY_METHOD.get(method)
    if field is None:
        return

    raw = headers.get(HEADER_NAME)
    if raw is None:
        raise HeaderMismatchError(
            f'Header mismatch: missing required {HEADER_NAME} header'
        )
    decoded = decode_header_value(raw)
    body_value = params.get(field)
    if decoded != body_value:
        raise HeaderMismatchError(
            f'Header mismatch: {HEADER_NAME} header value {decoded!r} does not match body value {body_value!r}'
        )


def _expect_header(headers: Mapping[str, str], name: str, body_value: Any) -> None:
    value = headers.get(name)
    if value is None:
        raise HeaderMismatchError(f'Header mismatch: missing required {name} header')
    if value != body_value:
        raise HeaderMismatchError(
            f'Header mismatch: {name} header value {value!r} does not match body value {body_value!r}'
        )


def decode_header_value(value: str) -> str:
    """Decode a `=?base64?...?=` sentinel header value; plain values pass through."""
    if not (value.startswith(_BASE64_PREFIX) and value.endswith(_BASE64_SUFFIX)):
        return value
    encoded = value[len(_BASE64_PREFIX) : -len(_BASE64_SUFFIX)]
    try:
        return base64.b64decode(encoded, validate=True).decode('utf-8')
    except (binascii.Error, UnicodeDecodeError) as e:
        raise HeaderMismatchError(
            f'Header mismatch: malformed Base64 header value {value!r}'
        ) from e


def is_origin_allowed(
    origin: str | None, host: str, allowed_origins: Iterable[str]
) -> bool:
    """Origin validation against DNS rebinding.

    An absent Origin (server-to-server traffic) is allowed. Otherwise the
    Origin's host must equal the request `Host`, or the Origin must be listed
    in `allowed_origins`.
    """
    if origin is None:
        return True
    normalized = _normalize_origin(origin)
    if normalized in {_normalize_origin(o) for o in allowed_origins}:
        return True

    parts = urlsplit(normalized)
    if not parts.scheme or not parts.hostname:
        return False
    return _strip_default_port(parts.netloc, parts.scheme) == _strip_default_port(
        host.lower(), parts.scheme
    )


def _normalize_origin(origin: str) -> str:
    return origin.strip().rstrip('/').lower()


def _strip_default_port(netloc: str, scheme: str) -> str:
    default = _DEFAULT_PORTS.get(scheme)
    if default is not None and netloc.endswith(f':{default}'):
        return netloc[: -len(f':{default}')]
    return netloc
