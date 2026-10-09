from __future__ import annotations

from frappe_mcp.server import protocol


def handle_initialize(params, server_info: dict, instructions: str | None = None):
    """
    Handles the legacy initialize request from the client.

    Echoes the client's requested version when it is a supported legacy version,
    otherwise answers with the latest legacy version.
    """
    requested = params.get('protocolVersion')
    if requested not in protocol.LEGACY_VERSIONS:
        requested = protocol.DEFAULT_LEGACY_VERSION

    result = {
        'protocolVersion': requested,
        'serverInfo': server_info,
        'capabilities': {
            'tools': {'listChanged': False},
            'prompts': {'listChanged': False},
            # Not yet implemented
            # "completions": {},
            # "resources": {"subscribe": True, "listChanged": False},
            # "logging": {},
        },
    }
    if instructions:
        result['instructions'] = instructions
    return result


def handle_discover(_params, instructions: str | None = None):
    """
    Handles the modern server/discover request.
    https://modelcontextprotocol.io/specification/2026-07-28/server/discover

    serverInfo, resultType and caching hints are added by the server.
    """
    result = {
        'supportedVersions': list(protocol.SUPPORTED_VERSIONS),
        'capabilities': {'tools': {}, 'prompts': {}},
    }
    if instructions:
        result['instructions'] = instructions
    return result


def handle_ping(_):
    """
    Handles the ping request from the client.
    https://modelcontextprotocol.io/specification/2025-03-26/basic/utilities/ping#ping
    """
    return {}
