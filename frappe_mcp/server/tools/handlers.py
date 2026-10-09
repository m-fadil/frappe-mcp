from __future__ import annotations

import json
import logging
from collections import OrderedDict
from typing import Any

from jsonschema import ValidationError as ArgumentsError
from pydantic import ValidationError

import frappe_mcp.server.tools as tools
from frappe_mcp.server import runtime, types

logger = logging.getLogger(__name__)


def handle_call_tool(params, tool_registry: OrderedDict[str, tools.Tool]):
    """
    Handles the tools/call request from the client.
    https://modelcontextprotocol.io/specification/2026-07-28/server/tools#error-handling

    Unknown tools are protocol errors (ValueError -> -32602). Invalid arguments
    and failures inside the tool are tool execution errors (`isError: true`);
    a failing tool's database writes are rolled back. Only messages meant for
    the caller (see `runtime.user_message`) reach the client; other errors are
    logged and reported generically.
    """
    call_params = types.CallToolRequestParams.model_validate(params)
    tool_name = call_params.name

    tool = tool_registry.get(tool_name)
    if tool is None:
        raise ValueError(f'Unknown tool: {tool_name}')

    try:
        tool_result = tools.run_tool(tool, call_params.arguments or {})
    except ArgumentsError as e:
        runtime.rollback()
        return _error_result(f"Invalid arguments for tool '{tool_name}': {e.message}")
    except Exception as e:
        runtime.rollback()
        if (message := runtime.user_message(e)) is not None:
            return _error_result(f"Error calling tool '{tool_name}': {message}")
        runtime.log_exception(f"MCP tool '{tool_name}' failed")
        return _error_result(
            f"Error calling tool '{tool_name}': internal error (logged on the server)"
        )
    return _to_call_result(tool_result)


def _error_result(text: str) -> dict:
    return types.dump(
        types.CallToolResult(content=[types.TextContent(text=text)], isError=True)
    )


def _to_call_result(tool_result: Any) -> dict:
    if isinstance(tool_result, str):
        text, structured = tool_result, None
    else:
        text = json.dumps(tool_result, default=runtime.json_default)
        parsed = json.loads(text)
        # Structured content stays an object for legacy clients.
        structured = parsed if isinstance(parsed, dict) else None

    return types.dump(
        types.CallToolResult(
            content=[types.TextContent(text=text)],
            structuredContent=structured,
            isError=False,
        )
    )


def handle_list_tools(params, tool_registry: OrderedDict[str, tools.Tool]):
    """
    Handles the tools/list request from the client.
    https://modelcontextprotocol.io/specification/2025-06-18/tools/list#toolslist
    """
    types.ListToolsRequestParams.model_validate(params)

    tool_list = []
    for tool_info in tool_registry.values():
        if tool := get_validated_tool(tool_info):
            tool_list.append(tool)

    return types.dump(types.ListToolsResult(tools=tool_list))


def get_validated_tool(tool: tools.Tool):
    t = {
        'name': tool.get('name'),
        'description': tool.get('description'),
        'inputSchema': tool.get('input_schema'),
        'outputSchema': tool.get('output_schema'),
        'annotations': tool.get('annotations'),
    }

    if t['outputSchema'] is None:
        del t['outputSchema']
    if t['annotations'] is None:
        del t['annotations']

    try:
        return types.Tool.model_validate(t)
    except ValidationError as e:
        logger.warning('Skipping invalid tool %r: %s', t['name'], e)
    return None
