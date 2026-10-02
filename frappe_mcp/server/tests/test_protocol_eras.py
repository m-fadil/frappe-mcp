import base64
import io
import json

import pytest
from werkzeug.wrappers import Request, Response

import frappe_mcp
from frappe_mcp.server import types
from frappe_mcp.server.server import MCP

MODERN = '2026-07-28'
SERVER_INFO_KEY = 'io.modelcontextprotocol/serverInfo'


@pytest.fixture
def mcp():
    mcp = MCP(name='era-mcp', instructions='Be nice.')

    @mcp.tool()
    def adder(a: int, b: int) -> dict:
        """Adds two numbers."""
        return {'output': a + b}

    @mcp.tool(name='héllo wörld')
    def greet():
        """Non-ASCII tool name."""
        return 'hi'

    @mcp.prompt()
    def summarize(topic: str):
        """Summarize a topic."""
        return [
            types.PromptMessage(
                role='user', content=types.TextContent(text=f'Summarize {topic}')
            )
        ]

    return mcp


def modern_params(params=None, version=MODERN, capabilities=None):
    meta = {
        'io.modelcontextprotocol/protocolVersion': version,
        'io.modelcontextprotocol/clientInfo': {'name': 'test', 'version': '1.0.0'},
        'io.modelcontextprotocol/clientCapabilities': {}
        if capabilities is None
        else capabilities,
    }
    return {**(params or {}), '_meta': meta}


def modern_headers(method, name=None, version=MODERN):
    headers = {'MCP-Protocol-Version': version, 'Mcp-Method': method}
    if name is not None:
        headers['Mcp-Name'] = name
    return headers


def post(mcp, method, params=None, headers=None, request_id=7, http_method='POST'):
    body = {'jsonrpc': '2.0', 'id': request_id, 'method': method}
    if params is not None:
        body['params'] = params
    request = Request.from_values(
        method=http_method,
        content_type='application/json',
        input_stream=io.BytesIO(json.dumps(body).encode()),
        headers=headers or {},
    )
    response = mcp.handle(request, Response())
    data = json.loads(response.data) if response.data else None
    return response.status_code, data


def modern(mcp, method, params=None, name=None, **kwargs):
    return post(
        mcp, method, modern_params(params), modern_headers(method, name), **kwargs
    )


# ---------------------------------------------------------------------------
# Modern requests
# ---------------------------------------------------------------------------


def test_discover(mcp):
    status, data = modern(mcp, 'server/discover')
    assert status == 200
    assert data['id'] == 7
    result = data['result']
    assert result['resultType'] == 'complete'
    assert result['supportedVersions'] == [
        '2026-07-28',
        '2025-11-25',
        '2025-06-18',
        '2025-03-26',
    ]
    assert result['capabilities'] == {'tools': {}, 'prompts': {}}
    assert result['_meta'][SERVER_INFO_KEY] == {
        'name': 'era-mcp',
        'version': frappe_mcp.__version__,
    }
    assert result['instructions'] == 'Be nice.'
    assert result['cacheScope'] == 'private'
    assert result['ttlMs'] == 60_000


@pytest.mark.parametrize('method', ['tools/list', 'prompts/list'])
def test_lists_carry_cache_hints(mcp, method):
    status, data = modern(mcp, method)
    assert status == 200
    result = data['result']
    assert result['resultType'] == 'complete'
    assert result['ttlMs'] == 60_000
    assert result['cacheScope'] == 'private'
    assert result['_meta'][SERVER_INFO_KEY]['version'] == frappe_mcp.__version__


def test_cache_ttl_is_configurable():
    mcp = MCP('ttl', cache_ttl_ms=0)
    _, data = modern(mcp, 'tools/list')
    assert data['result']['ttlMs'] == 0


def test_negative_cache_ttl_rejected():
    with pytest.raises(ValueError):
        MCP('ttl', cache_ttl_ms=-1)


def test_tools_list_order_is_registration_order(mcp):
    _, data = modern(mcp, 'tools/list')
    assert [t['name'] for t in data['result']['tools']] == ['adder', 'héllo wörld']


def test_tools_call_result_is_complete_without_cache_hints(mcp):
    status, data = modern(
        mcp, 'tools/call', {'name': 'adder', 'arguments': {'a': 2, 'b': 3}}, 'adder'
    )
    assert status == 200
    result = data['result']
    assert result['structuredContent'] == {'output': 5}
    assert result['resultType'] == 'complete'
    assert SERVER_INFO_KEY in result['_meta']
    assert 'ttlMs' not in result
    assert 'cacheScope' not in result


def test_tools_call_uses_swapped_registry(mcp):
    other = MCP('other')

    @other.tool()
    def only_here():
        """Only in the swapped registry."""
        return 'swapped'

    mcp._tool_registry = other._tool_registry
    _, data = modern(mcp, 'tools/call', {'name': 'only_here'}, 'only_here')
    assert data['result']['content'] == [{'type': 'text', 'text': 'swapped'}]


def test_prompts_get(mcp):
    status, data = modern(
        mcp,
        'prompts/get',
        {'name': 'summarize', 'arguments': {'topic': 'Go'}},
        'summarize',
    )
    assert status == 200
    assert data['result']['resultType'] == 'complete'
    assert 'Go' in data['result']['messages'][0]['content']['text']


def test_base64_mcp_name(mcp):
    encoded = base64.b64encode('héllo wörld'.encode()).decode()
    status, data = modern(
        mcp, 'tools/call', {'name': 'héllo wörld'}, f'=?base64?{encoded}?='
    )
    assert status == 200
    assert data['result']['content'] == [{'type': 'text', 'text': 'hi'}]


def test_base64_mcp_name_mismatch(mcp):
    encoded = base64.b64encode(b'subtractor').decode()
    status, data = modern(mcp, 'tools/call', {'name': 'adder'}, f'=?base64?{encoded}?=')
    assert status == 400
    assert data['error']['code'] == -32020


def test_malformed_base64_mcp_name(mcp):
    status, data = modern(mcp, 'tools/call', {'name': 'adder'}, '=?base64?***?=')
    assert status == 400
    assert data['error']['code'] == -32020


def test_unsupported_version(mcp):
    status, data = post(
        mcp,
        'tools/list',
        modern_params(version='1900-01-01'),
        modern_headers('tools/list', version='1900-01-01'),
    )
    assert status == 400
    assert data['id'] == 7
    assert data['error']['code'] == -32022
    assert data['error']['message'] == 'Unsupported protocol version'
    assert data['error']['data'] == {
        'supported': ['2026-07-28', '2025-11-25', '2025-06-18', '2025-03-26'],
        'requested': '1900-01-01',
    }


def test_legacy_version_in_meta_is_unsupported(mcp):
    # Legacy versions are reachable via `initialize`, not per-request _meta.
    status, data = post(
        mcp,
        'tools/list',
        modern_params(version='2025-11-25'),
        modern_headers('tools/list', version='2025-11-25'),
    )
    assert status == 400
    assert data['error']['code'] == -32022


def test_missing_client_capabilities(mcp):
    params = modern_params()
    del params['_meta']['io.modelcontextprotocol/clientCapabilities']
    status, data = post(mcp, 'tools/list', params, modern_headers('tools/list'))
    assert status == 400
    assert data['error']['code'] == -32602


@pytest.mark.parametrize(
    ('headers', 'fragment'),
    [
        ({'Mcp-Method': 'tools/list'}, 'MCP-Protocol-Version'),
        (
            {'MCP-Protocol-Version': '2025-11-25', 'Mcp-Method': 'tools/list'},
            'MCP-Protocol-Version',
        ),
        ({'MCP-Protocol-Version': MODERN}, 'Mcp-Method'),
        ({'MCP-Protocol-Version': MODERN, 'Mcp-Method': 'tools/call'}, 'Mcp-Method'),
    ],
)
def test_header_mismatch(mcp, headers, fragment):
    status, data = post(mcp, 'tools/list', modern_params(), headers)
    assert status == 400
    assert data['id'] == 7
    assert data['error']['code'] == -32020
    assert fragment in data['error']['message']


def test_missing_mcp_name(mcp):
    status, data = modern(mcp, 'tools/call', {'name': 'adder'})
    assert status == 400
    assert data['error']['code'] == -32020
    assert 'Mcp-Name' in data['error']['message']


def test_mcp_name_mismatch(mcp):
    status, data = modern(mcp, 'prompts/get', {'name': 'summarize'}, 'other')
    assert status == 400
    assert data['error']['code'] == -32020


def test_resources_read_requires_mcp_name_for_uri(mcp):
    status, data = modern(mcp, 'resources/read', {'uri': 'file:///a'})
    assert status == 400
    assert data['error']['code'] == -32020
    status, data = modern(mcp, 'resources/read', {'uri': 'file:///a'}, 'file:///a')
    assert status == 404
    assert data['error']['code'] == -32601


def test_header_names_are_case_insensitive(mcp):
    headers = {
        'mcp-protocol-version': MODERN,
        'MCP-METHOD': 'tools/call',
        'mcp-name': 'adder',
    }
    status, data = post(
        mcp,
        'tools/call',
        modern_params({'name': 'adder', 'arguments': {'a': 1, 'b': 1}}),
        headers,
    )
    assert status == 200
    assert data['result']['structuredContent'] == {'output': 2}


@pytest.mark.parametrize(
    'method',
    [
        'initialize',
        'ping',
        'logging/setLevel',
        'resources/subscribe',
        'resources/unsubscribe',
        'foo/bar',
        'completion/complete',
    ],
)
def test_modern_method_not_found(mcp, method):
    status, data = modern(mcp, method)
    assert status == 404
    assert data['id'] == 7
    assert data['error']['code'] == -32601


# ---------------------------------------------------------------------------
# Legacy requests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('version', ['2025-11-25', '2025-06-18', '2025-03-26'])
def test_legacy_initialize_echoes_supported_version(mcp, version):
    status, data = post(
        mcp,
        'initialize',
        {
            'protocolVersion': version,
            'capabilities': {},
            'clientInfo': {'name': 'c', 'version': '1'},
        },
    )
    assert status == 200
    result = data['result']
    assert result['protocolVersion'] == version
    assert result['serverInfo'] == {
        'name': 'era-mcp',
        'version': frappe_mcp.__version__,
    }
    assert 'resultType' not in result


@pytest.mark.parametrize('version', ['2024-11-05', '2026-07-28', None])
def test_legacy_initialize_falls_back_to_latest_legacy(mcp, version):
    params = {'capabilities': {}, 'clientInfo': {'name': 'c', 'version': '1'}}
    if version:
        params['protocolVersion'] = version
    _, data = post(mcp, 'initialize', params)
    assert data['result']['protocolVersion'] == '2025-11-25'


def test_legacy_tools_call_without_meta_or_headers(mcp):
    status, data = post(
        mcp, 'tools/call', {'name': 'adder', 'arguments': {'a': 1, 'b': 2}}
    )
    assert status == 200
    result = data['result']
    assert result['structuredContent'] == {'output': 3}
    assert 'resultType' not in result
    assert '_meta' not in result


def test_legacy_list_has_no_cache_hints(mcp):
    _, data = post(mcp, 'tools/list', {})
    assert set(data['result']) == {'tools'}


def test_legacy_unknown_method_keeps_400(mcp):
    status, data = post(mcp, 'server/discover', {})
    assert status == 400
    assert data['error']['code'] == -32601


def test_legacy_ping(mcp):
    status, data = post(mcp, 'ping')
    assert status == 200
    assert data['result'] == {}


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def test_foreign_origin_rejected(mcp):
    status, data = post(mcp, 'tools/list', {}, {'Origin': 'https://evil.example'})
    assert status == 403
    assert 'id' not in data
    assert data['error']['code'] == -32600


def test_foreign_origin_rejected_for_get(mcp):
    status, _ = post(
        mcp, 'tools/list', {}, {'Origin': 'https://evil.example'}, http_method='GET'
    )
    assert status == 403


@pytest.mark.parametrize(
    ('origin', 'host'),
    [
        ('http://localhost', 'localhost'),
        ('https://site.example:8443', 'site.example:8443'),
        ('https://Site.Example', 'site.example:443'),
    ],
)
def test_same_host_origin_allowed(mcp, origin, host):
    status, _ = post(mcp, 'tools/list', {}, {'Origin': origin, 'Host': host})
    assert status == 200


def test_origin_port_must_match(mcp):
    status, _ = post(
        mcp, 'tools/list', {}, {'Origin': 'http://localhost:9999', 'Host': 'localhost'}
    )
    assert status == 403


def test_allowed_origin():
    mcp = MCP('o', allowed_origins=['https://app.example.com/'])
    status, _ = post(mcp, 'tools/list', {}, {'Origin': 'https://app.example.com'})
    assert status == 200
    status, _ = post(mcp, 'tools/list', {}, {'Origin': 'null'})
    assert status == 403


def test_absent_origin_allowed(mcp):
    status, _ = post(mcp, 'tools/list', {})
    assert status == 200


@pytest.mark.parametrize('http_method', ['GET', 'DELETE'])
def test_get_delete_not_allowed(mcp, http_method):
    status, _ = post(mcp, 'tools/list', {}, http_method=http_method)
    assert status == 405


def test_modern_notification_accepted(mcp):
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(
            json.dumps(
                {
                    'jsonrpc': '2.0',
                    'method': 'notifications/cancelled',
                    'params': modern_params(),
                }
            ).encode()
        ),
    )
    assert mcp.handle(request, Response()).status_code == 202
