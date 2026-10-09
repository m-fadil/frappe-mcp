import datetime
import decimal
import io
import json
from collections import OrderedDict

import pytest
from werkzeug.wrappers import Request, Response

from frappe_mcp import ToolError
from frappe_mcp.server import runtime, types
from frappe_mcp.server.server import MCP


@pytest.fixture
def mcp_instance():
    mcp = MCP(name='frappe-mcp')

    @mcp.tool()
    def adder(a: int, b: int) -> dict:
        """Adds two numbers."""
        return {'output': a + b}

    @mcp.tool()
    def subtractor(a: int, b: int):
        """Subtracts two numbers."""
        return a - b

    return mcp


@pytest.fixture
def mcp_with_prompts():
    mcp = MCP(name='frappe-mcp')

    @mcp.prompt()
    def summarize(topic: str, language: str = 'english'):
        """Summarize a topic."""
        return [
            types.PromptMessage(
                role='user',
                content=types.TextContent(text=f'Summarize {topic} in {language}'),
            )
        ]

    @mcp.prompt()
    def no_args_prompt():
        """A prompt with no arguments."""
        return [
            types.PromptMessage(role='user', content=types.TextContent(text='Hello'))
        ]

    return mcp


def test_handle_initialize(mcp_instance):
    request_data = {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {'clientInfo': {'name': 'test-client'}},
    }
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(json.dumps(request_data).encode('utf-8')),
    )

    response = mcp_instance.handle(request, Response())

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['id'] == 1
    assert 'result' in response_data
    assert 'serverInfo' in response_data['result']
    assert response_data['result']['serverInfo']['name'] == 'frappe-mcp'


def test_handle_initialized_notification(mcp_instance):
    request_data = {
        'jsonrpc': '2.0',
        'method': 'notifications/initialized',
        'params': {},
    }
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(json.dumps(request_data).encode('utf-8')),
    )

    response = mcp_instance.handle(request, Response())

    assert response.status_code == 202
    assert not response.data


def test_handle_list_tools(mcp_instance):
    request_data = {'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list', 'params': {}}
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(json.dumps(request_data).encode('utf-8')),
    )

    response = mcp_instance.handle(request, Response())

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['id'] == 2
    assert 'result' in response_data
    assert 'tools' in response_data['result']
    tools = response_data['result']['tools']
    assert len(tools) == 2
    tool_names = {tool['name'] for tool in tools}
    assert tool_names == {'adder', 'subtractor'}


def test_handle_call_tool_with_structured_content(mcp_instance):
    request_data = {
        'jsonrpc': '2.0',
        'id': 3,
        'method': 'tools/call',
        'params': {'name': 'adder', 'arguments': {'a': 5, 'b': 10}},
    }
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(json.dumps(request_data).encode('utf-8')),
    )

    response = mcp_instance.handle(request, Response())

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['id'] == 3
    assert 'result' in response_data
    assert response_data['result']['structuredContent'] == {'output': 15}


def test_handle_call_tool_with_regular_content(mcp_instance):
    request_data = {
        'jsonrpc': '2.0',
        'id': 3,
        'method': 'tools/call',
        'params': {'name': 'subtractor', 'arguments': {'a': 7, 'b': 4}},
    }
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(json.dumps(request_data).encode('utf-8')),
    )

    response = mcp_instance.handle(request, Response())

    assert response.status_code == 200
    response_data = json.loads(response.data)
    assert response_data['id'] == 3
    assert 'result' in response_data
    # assert response_data['result']['structuredContent'] is None
    assert response_data['result']['content'] == [{'type': 'text', 'text': '3'}]


def _post(mcp, method, params=None, request_id=1):
    data = {
        'jsonrpc': '2.0',
        'id': request_id,
        'method': method,
        'params': params or {},
    }
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(json.dumps(data).encode('utf-8')),
    )
    return json.loads(mcp.handle(request, Response()).data)


def test_initialize_has_prompts_capability(mcp_instance):
    result = _post(mcp_instance, 'initialize', {'clientInfo': {'name': 'test'}})
    assert 'prompts' in result['result']['capabilities']


def test_handle_list_prompts_empty(mcp_instance):
    result = _post(mcp_instance, 'prompts/list')
    assert result['result'] == {'prompts': []}


def test_handle_list_prompts(mcp_with_prompts):
    result = _post(mcp_with_prompts, 'prompts/list')
    prompts = result['result']['prompts']
    assert len(prompts) == 2
    names = {p['name'] for p in prompts}
    assert names == {'summarize', 'no_args_prompt'}


def test_handle_list_prompts_arguments(mcp_with_prompts):
    result = _post(mcp_with_prompts, 'prompts/list')
    summarize = next(p for p in result['result']['prompts'] if p['name'] == 'summarize')
    args = {a['name']: a for a in summarize['arguments']}
    assert args['topic']['required'] is True
    assert args['language']['required'] is False


def test_handle_get_prompt(mcp_with_prompts):
    result = _post(
        mcp_with_prompts,
        'prompts/get',
        {'name': 'summarize', 'arguments': {'topic': 'Python'}},
    )
    messages = result['result']['messages']
    assert len(messages) == 1
    assert messages[0]['role'] == 'user'
    assert 'Python' in messages[0]['content']['text']


def test_handle_get_prompt_default_arg(mcp_with_prompts):
    result = _post(
        mcp_with_prompts,
        'prompts/get',
        {'name': 'summarize', 'arguments': {'topic': 'Go', 'language': 'french'}},
    )
    assert 'french' in result['result']['messages'][0]['content']['text']


def test_handle_get_prompt_not_found(mcp_with_prompts):
    result = _post(mcp_with_prompts, 'prompts/get', {'name': 'nonexistent'})
    assert result.get('error') is not None
    assert result['error']['code'] == -32602


def test_unimplemented_method_returns_error(mcp_instance):
    result = _post(mcp_instance, 'completion/complete', {'ref': {}, 'argument': {}})
    assert result.get('error') is not None
    assert result['error']['code'] == -32601


def test_unknown_method_returns_error(mcp_instance):
    result = _post(mcp_instance, 'foo/bar')
    assert result.get('error') is not None
    assert result['error']['code'] == -32601


def test_handler_exception_returns_internal_error(mcp_instance):
    @mcp_instance.tool()
    def boom():
        """Raises unexpectedly."""
        raise RuntimeError('something went wrong')

    result = _post(mcp_instance, 'tools/call', {'name': 'boom', 'arguments': {}})
    assert result['result']['isError'] is True


def _post_raw(mcp, body: bytes):
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(body),
    )
    response = mcp.handle(request, Response())
    return response.status_code, json.loads(response.data)


@pytest.mark.parametrize(
    ('body', 'code'),
    [
        (b'{bad', -32700),
        (b'\xff\xfe', -32700),
        (b'[{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}]', -32600),
        (b'"tools/list"', -32600),
        (b'null', -32600),
        (b'{"jsonrpc": "2.0", "id": 1}', -32600),
        (b'{"jsonrpc": "1.0", "id": 1, "method": "tools/list"}', -32600),
        (b'{"jsonrpc": "2.0", "id": true, "method": "tools/list"}', -32600),
        (b'{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": []}', -32600),
    ],
)
def test_malformed_body_returns_jsonrpc_error(mcp_instance, body, code):
    status, data = _post_raw(mcp_instance, body)
    assert status == 400
    assert data['error']['code'] == code


def test_call_tool_with_invalid_arguments_is_tool_error(mcp_instance):
    result = _post(
        mcp_instance, 'tools/call', {'name': 'adder', 'arguments': {'a': 'x', 'b': 1}}
    )
    assert result['result']['isError'] is True
    assert (
        "Invalid arguments for tool 'adder'" in result['result']['content'][0]['text']
    )


def test_call_tool_with_missing_argument_is_tool_error(mcp_instance):
    result = _post(mcp_instance, 'tools/call', {'name': 'adder', 'arguments': {'a': 1}})
    assert result['result']['isError'] is True


def test_call_tool_drops_undeclared_arguments(mcp_instance):
    result = _post(
        mcp_instance,
        'tools/call',
        {'name': 'adder', 'arguments': {'a': 1, 'b': 2, 'extra': True}},
    )
    assert result['result']['structuredContent'] == {'output': 3}


def test_call_unknown_tool_is_protocol_error(mcp_instance):
    result = _post(mcp_instance, 'tools/call', {'name': 'missing'})
    assert result['error']['code'] == -32602
    assert result['error']['message'] == 'Unknown tool: missing'


def test_error_id_is_echoed_only_when_valid():
    mcp = MCP('x')
    _, data = _post_raw(mcp, b'{"jsonrpc": "1.0", "id": "a", "method": "tools/list"}')
    assert data['id'] == 'a'
    _, data = _post_raw(mcp, b'{"jsonrpc": "2.0", "id": true, "method": "tools/list"}')
    assert data.get('id') is None


def test_tool_result_with_dates_and_decimals_is_json():
    mcp = MCP('x')

    @mcp.tool()
    def invoice():
        return {
            'posting_date': datetime.date(2026, 10, 9),
            'amount': decimal.Decimal('1.50'),
        }

    result = _post(mcp, 'tools/call', {'name': 'invoice'})['result']
    expected = {'posting_date': '2026-10-09', 'amount': 1.5}
    assert result['structuredContent'] == expected
    assert json.loads(result['content'][0]['text']) == expected


def test_unexpected_tool_exception_is_not_leaked(caplog):
    mcp = MCP('x')

    @mcp.tool()
    def leak():
        raise RuntimeError('pymysql: secret table at /srv/app')

    result = _post(mcp, 'tools/call', {'name': 'leak'})['result']
    text = result['content'][0]['text']
    assert result['isError'] is True
    assert 'secret' not in text
    assert 'internal error' in text
    assert 'secret table' in caplog.text


def test_tool_error_message_reaches_client():
    mcp = MCP('x')

    @mcp.tool()
    def refuse():
        raise ToolError('Customer is on hold')

    result = _post(mcp, 'tools/call', {'name': 'refuse'})['result']
    assert result['isError'] is True
    assert 'Customer is on hold' in result['content'][0]['text']


def test_unexpected_handler_exception_is_not_leaked():
    mcp = MCP('x')

    @mcp.prompt()
    def broken():
        raise RuntimeError('secret')

    result = _post(mcp, 'prompts/get', {'name': 'broken'})
    assert result['error']['code'] == -32603
    assert 'secret' not in result['error']['message']


def test_prompt_missing_required_argument_is_invalid_params(mcp_with_prompts):
    result = _post(mcp_with_prompts, 'prompts/get', {'name': 'summarize'})
    assert result['error']['code'] == -32602
    assert 'topic' in result['error']['message']


class _FakeDB:
    def __init__(self):
        self.calls = []

    def savepoint(self, name):
        self.calls.append(('savepoint', name))

    def rollback(self, *, save_point=None):
        self.calls.append(('rollback', save_point))


@pytest.fixture
def fake_db(monkeypatch):
    db = _FakeDB()
    monkeypatch.setattr(runtime, '_db', lambda: db)
    return db


def test_failing_tool_rolls_back_to_savepoint(fake_db):
    mcp = MCP('x')

    @mcp.tool()
    def fails():
        raise RuntimeError('boom')

    _post(mcp, 'tools/call', {'name': 'fails'})
    assert fake_db.calls == [
        ('savepoint', 'frappe_mcp_request'),
        ('rollback', 'frappe_mcp_request'),
    ]


def test_successful_tool_keeps_its_writes(fake_db):
    mcp = MCP('x')

    @mcp.tool()
    def works():
        return 'ok'

    _post(mcp, 'tools/call', {'name': 'works'})
    assert ('rollback', 'frappe_mcp_request') not in fake_db.calls


def test_rollback_falls_back_when_savepoint_is_gone(fake_db, monkeypatch):
    def rollback(*, save_point=None):
        fake_db.calls.append(('rollback', save_point))
        if save_point:
            raise RuntimeError('SAVEPOINT does not exist')

    monkeypatch.setattr(fake_db, 'rollback', rollback)
    with runtime.request_scope():
        runtime.rollback()
    assert fake_db.calls == [
        ('savepoint', 'frappe_mcp_request'),
        ('rollback', 'frappe_mcp_request'),
        ('rollback', None),
    ]


def test_handler_called_directly_leaves_transaction_to_caller(fake_db):
    from frappe_mcp.server.tools import get_tool, handle_call_tool

    def fails():
        raise RuntimeError('boom')

    result = handle_call_tool({'name': 'fails'}, {'fails': get_tool(fails)})
    assert result['isError'] is True
    assert fake_db.calls == []


def test_per_request_tool_registry_does_not_touch_instance(mcp_instance):
    only_adder = OrderedDict(adder=mcp_instance._tool_registry['adder'])
    request = Request.from_values(
        method='POST',
        content_type='application/json',
        input_stream=io.BytesIO(
            json.dumps(
                {'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list', 'params': {}}
            ).encode()
        ),
    )
    data = json.loads(
        mcp_instance.handle(request, Response(), tool_registry=only_adder).data
    )
    assert [t['name'] for t in data['result']['tools']] == ['adder']
    assert set(mcp_instance._tool_registry) == {'adder', 'subtractor'}
