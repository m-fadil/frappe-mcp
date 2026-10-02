from __future__ import annotations

import json
from collections import OrderedDict
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel, ValidationError
from werkzeug.wrappers import Request, Response

import frappe_mcp.server.handlers as handlers
import frappe_mcp.server.prompts as prompts
import frappe_mcp.server.tools as tools
from frappe_mcp.server import protocol, types

__all__ = ['MCP']


class MCP:
    """The main class for creating an MCP server.

    This class orchestrates the handling of JSON-RPC requests, manages a
    registry of available tools, and integrates with a web server framework
    to expose the MCP functionality.

    In a Frappe application, you would typically create a single instance of
    this class and use the `@mcp.register()` decorator on an API endpoint.
    Tools can be added using the `@mcp.tool()` decorator.

    Example:
        ```python
        # In app/mcp.py
        from frappe_mcp import MCP

        mcp = MCP(name="my-mcp-server")

        @mcp.tool()
        def my_tool(param1: str):
            '''A simple tool.'''
            return f"You said: {param1}"

        @mcp.register()
        def mcp_endpoint():
            '''The entry point for MCP requests.'''
            # This function body is executed before request handling.
            # It's a good place to import modules that register tools.
            pass
        ```

    For use in other Werkzeug-based servers, you can use the `mcp.handle()`
    method directly.

    Args:
        name: The server name reported in serverInfo. Defaults to "frappe-mcp".
        allowed_origins: Extra browser origins (e.g. "https://app.example.com")
            allowed to call the endpoint. A request whose `Origin` header is
            present, differs from the request host and is not listed here is
            rejected with HTTP 403.
        instructions: Optional natural-language guidance for LLMs, returned by
            `server/discover` and legacy `initialize`.
        cache_ttl_ms: `ttlMs` freshness hint on modern `server/discover`,
            `tools/list` and `prompts/list` results. These results are always
            marked `cacheScope: "private"` because Frappe catalogues can vary
            per caller and per site settings.
    """

    _name: str | None
    _tool_registry: OrderedDict[str, tools.Tool]
    _prompt_registry: OrderedDict[str, prompts.Prompt]
    _mcp_entry_fn: Callable | None
    _allowed_origins: tuple[str, ...]
    _instructions: str | None
    _cache_ttl_ms: int

    def __init__(
        self,
        name: str | None,
        *,
        allowed_origins: Iterable[str] = (),
        instructions: str | None = None,
        cache_ttl_ms: int = protocol.DEFAULT_CACHE_TTL_MS,
    ):
        if cache_ttl_ms < 0:
            raise ValueError('cache_ttl_ms must be >= 0')
        self._tool_registry = OrderedDict()
        self._prompt_registry = OrderedDict()
        self._name = name
        self._mcp_entry_fn = None
        self._allowed_origins = tuple(allowed_origins)
        self._instructions = instructions
        self._cache_ttl_ms = cache_ttl_ms

    def register(
        self,
        *,
        allow_guest: bool = False,
        xss_safe: bool = False,
    ):
        """A decorator to mark a function as an MCP endpoint.

        This is a wrapper around frappe.whitelist() that sets up the necessary
        configuration for handling MCP requests. The decorated function will be
        used as the entry point for all MCP requests.

        Only one function can be registered as an MCP endpoint per MCP instance.

        Args:
            allow_guest: If True, allows unauthenticated access to the endpoint.
            xss_safe: If True, response will not be sanitized for XSS.

        Raises:
            Exception: If not used in a Frappe app, or if already registered.
        """
        from werkzeug import Response

        try:
            import frappe
        except ImportError as e:
            raise Exception(
                'mcp.register can be used only in a Frappe app.\n'
                'If you are using it in some other Werkzeug based server\n'
                'you should use the mcp.handle function instead.'
            ) from e

        whitelister = frappe.whitelist(
            allow_guest=allow_guest,
            xss_safe=xss_safe,
            methods=['GET', 'POST'],
        )

        def decorator(fn):
            if self._mcp_entry_fn is not None:
                raise Exception('mcp.register can be used only once per MCP instance')

            self._mcp_entry_fn = fn

            def wrapper() -> Response:
                # Runs wrapped dummy mcp handler before handling the request.
                # This should import all the files with the registered mcp
                # functions.
                fn()

                request = frappe.request
                response = Response()

                return self.handle(request, response)

            return whitelister(wrapper)

        return decorator

    def handle(self, request: Request, response: Response) -> Response:
        """Handle an MCP request in any Werkzeug based server.

        This method can be used directly to integrate MCP functionality into any Werkzeug based server.
        It processes the request according to the MCP specification and returns an appropriate response.

        Args:
            request: The Werkzeug Request object containing the MCP request
            response: A Werkzeug Response object to be populated with the MCP response

        Returns:
            The populated Werkzeug Response object
        """
        if not protocol.is_origin_allowed(
            request.headers.get('Origin'), request.host, self._allowed_origins
        ):
            return handle_invalid(
                None,
                response,
                types.INVALID_REQUEST,
                'Forbidden: Origin not allowed',
                status=403,
            )

        if request.method != 'POST':
            response.status_code = 405
            return response

        try:
            data = request.get_json(force=True)
        except json.JSONDecodeError:
            return handle_invalid(None, response, types.PARSE_ERROR, 'Parse error')

        if get_is_notification(data):
            return handle_notification(data, response)

        if (request_id := data.get('id')) is None:
            return handle_invalid(
                request_id,
                response,
                types.INVALID_REQUEST,
                'Invalid Request',
            )

        return self._handle_request(request, request_id, data, response)

    def tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        input_schema: dict | None = None,
        use_entire_docstring: bool = False,
        annotations: tools.ToolAnnotations | None = None,
        # stream: bool = False,  # stream yes or no (SSE)
        # whitelist: list | None = None,
        # role: str | None = None,
    ):
        """A decorator that registers a function as a tool that can be used by an LLM.

        Example:
            >>> @mcp.tool()
            ... def get_current_weather(location: str, unit: str = "celsius"):
            ...     '''Get the current weather in a given location.'''
            ...     # ... implementation ...

        Args:
            name: The name of the tool. If not provided, the function's `__name__` will be used.
            description: A description of what the tool does. If not provided, it will be
                extracted from the function's docstring.
            input_schema: The JSON schema for the tool's input. If not provided, it will be
                inferred from the function's signature and docstring.
            use_entire_docstring: If True, the entire docstring will be used as the tool's
                description. Otherwise, only the first section is used (i.e. no Args).
            annotations: Additional context about the tool, such as validation information
                or examples of how to use it.
        """

        def decorator(fn: Callable):
            tool = tools.get_tool(
                fn,
                tools.ToolOptions(
                    name=name,
                    description=description,
                    input_schema=input_schema,
                    use_entire_docstring=use_entire_docstring,
                    annotations=annotations,
                ),
            )
            self.add_tool(tool)
            return fn

        return decorator

    def add_tool(self, tool: tools.Tool):
        """Registers a tool with the MCP instance.

        This method allows for adding a tool programmatically, serving as an
        alternative to the `@mcp.tool` decorator. The provided tool should
        be a dictionary conforming to the `frappe_mcp.Tool` `TypedDict` structure.

        Args:
            tool: The tool to register. It must be a dictionary with keys
                'name', 'description', 'input_schema', and 'fn'.
        """
        self._tool_registry[tool['name']] = tool

    def prompt(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        arguments: list[prompts.PromptArgument] | None = None,
    ):
        """A decorator that registers a function as a prompt template.

        The decorated function is called with string keyword arguments matching
        the declared prompt arguments and must return either a list of
        PromptMessage objects or a GetPromptResult instance.

        Example:
            >>> @mcp.prompt()
            ... def summarize(topic: str, language: str = "english"):
            ...     '''Summarize a topic.'''
            ...     return [PromptMessage(role="user", content=TextContent(text=f"Summarize {topic} in {language}"))]

        Args:
            name: The prompt name. Defaults to the function's __name__.
            description: A description of the prompt. Defaults to the docstring.
            arguments: Explicit argument list. Inferred from the signature if omitted.
        """

        def decorator(fn: Callable):
            prompt = prompts.get_prompt(
                fn,
                prompts.PromptOptions(
                    name=name,
                    description=description,
                    arguments=arguments,
                ),
            )
            self.add_prompt(prompt)
            return fn

        return decorator

    def add_prompt(self, prompt: prompts.Prompt):
        """Registers a prompt with the MCP instance.

        Args:
            prompt: A Prompt TypedDict with keys 'name', 'description',
                'arguments', and 'fn'.
        """
        self._prompt_registry[prompt['name']] = prompt

    def _server_info(self) -> dict:
        from frappe_mcp import __version__

        return {'name': self._name or 'frappe-mcp', 'version': __version__}

    def _handle_request(
        self,
        request: Request,
        request_id: types.RequestId,
        data: dict,
        response: Response,
    ) -> Response:
        # Request
        try:
            rpc_request = types.JSONRPCRequest.model_validate(data)
        except ValidationError as e:
            return handle_invalid(
                request_id,
                response,
                types.INVALID_PARAMS,
                f'Invalid params: {e}',
            )

        method = rpc_request.method
        params = rpc_request.params or {}

        requested_version = protocol.get_requested_version(params)
        modern = requested_version is not None
        if modern:
            if requested_version not in protocol.MODERN_VERSIONS:
                return handle_invalid(
                    request_id,
                    response,
                    types.UNSUPPORTED_PROTOCOL_VERSION,
                    'Unsupported protocol version',
                    data={
                        'supported': list(protocol.SUPPORTED_VERSIONS),
                        'requested': requested_version,
                    },
                )
            if not isinstance(
                params['_meta'].get(protocol.META_CLIENT_CAPABILITIES), dict
            ):
                return handle_invalid(
                    request_id,
                    response,
                    types.INVALID_PARAMS,
                    f'Invalid params: _meta is missing required {protocol.META_CLIENT_CAPABILITIES}',
                )
            try:
                protocol.validate_headers(request.headers, method, params)
            except protocol.HeaderMismatchError as e:
                return handle_invalid(
                    request_id, response, types.HEADER_MISMATCH, str(e)
                )

        # Modern servers answer unknown methods with HTTP 404; legacy keeps 400.
        not_found_status = 404 if modern else 400
        handler = self._get_handler(method, modern)
        if handler is None:
            return handle_invalid(
                request_id,
                response,
                types.METHOD_NOT_FOUND,
                'Method not found',
                status=not_found_status,
            )

        try:
            result = handler(params)
        except ValueError as e:
            return handle_invalid(request_id, response, types.INVALID_PARAMS, str(e))
        except NotImplementedError:
            return handle_invalid(
                request_id,
                response,
                types.METHOD_NOT_FOUND,
                'Method not implemented',
                status=not_found_status,
            )
        except Exception as e:
            return handle_invalid(
                request_id, response, types.INTERNAL_ERROR, f'Internal error: {e}'
            )

        result = {} if result is None else result
        if modern:
            result = self._complete_modern_result(method, result)
        success_response = types.JSONRPCSuccessResponse(id=request_id, result=result)
        response.data = get_response_data(success_response)
        response.mimetype = 'application/json'
        response.status_code = 200
        return response

    def _get_handler(
        self, method: str, modern: bool
    ) -> Callable[[dict], dict | None] | None:
        """Resolve the handler of `method` for the request's protocol era.

        Registries are read at call time because apps may swap
        `_tool_registry` per request.
        """
        match method:
            # Legacy only: the modern era has no handshake, ping, logging
            # level or resource subscriptions.
            case 'initialize' if not modern:
                return lambda p: handlers.handle_initialize(
                    p, self._server_info(), self._instructions
                )
            case 'ping' if not modern:
                return handlers.handle_ping
            case 'logging/setLevel' if not modern:
                return handlers.handle_set_level
            case 'resources/subscribe' if not modern:
                return handlers.handle_subscribe
            case 'resources/unsubscribe' if not modern:
                return handlers.handle_unsubscribe
            # Modern only.
            case 'server/discover' if modern:
                return lambda p: handlers.handle_discover(p, self._instructions)
            # Both eras.
            case 'completion/complete':
                return handlers.handle_complete
            case 'prompts/get':
                return lambda p: prompts.handle_get_prompt(p, self._prompt_registry)
            case 'prompts/list':
                return lambda p: prompts.handle_list_prompts(p, self._prompt_registry)
            case 'resources/list':
                return handlers.handle_list_resources
            case 'resources/templates/list':
                return handlers.handle_list_resource_templates
            case 'resources/read':
                return handlers.handle_read_resource
            case 'tools/call':
                return lambda p: tools.handle_call_tool(p, self._tool_registry)
            case 'tools/list':
                return lambda p: tools.handle_list_tools(p, self._tool_registry)
        return None

    def _complete_modern_result(self, method: str, result: dict) -> dict:
        meta = dict(result.get('_meta') or {})
        meta[protocol.META_SERVER_INFO] = self._server_info()
        result = {'resultType': 'complete', **result, '_meta': meta}
        if method in protocol.CACHEABLE_METHODS:
            result['ttlMs'] = self._cache_ttl_ms
            result['cacheScope'] = protocol.CACHE_SCOPE
        return result


def handle_notification(data: dict, response: Response) -> Response:
    # Notification
    try:
        rpc_notification = types.JSONRPCNotification.model_validate(data)
    except ValidationError:
        # Notifications with invalid params are ignored
        pass
    else:
        method = rpc_notification.method
        params = rpc_notification.params or {}
        match method:
            case 'notifications/cancelled':
                handlers.handle_cancelled(params)
            case 'notifications/progress':
                handlers.handle_progress(params)
            case 'notifications/initialized':
                handlers.handle_initialized(params)
            case 'notifications/roots/list_changed':
                handlers.handle_roots_list_changed(params)

    response.status_code = 202  # Accepted
    return response


def handle_invalid(
    request_id: types.RequestId,
    response: Response,
    code: int,
    message: str,
    *,
    data: Any = None,
    status: int = 400,
) -> Response:
    error_response = types.JSONRPCErrorResponse(
        id=request_id,
        error=types.Error(code=code, message=message, data=data),
    )
    response.data = get_response_data(error_response)
    response.mimetype = 'application/json'
    response.status_code = status
    return response


def get_response_data(model: BaseModel):
    return model.model_dump_json(exclude_none=True, by_alias=True)


def get_is_notification(data: dict) -> bool:
    method = data.get('method', '')
    return isinstance(method, str) and method.startswith('notifications/')
