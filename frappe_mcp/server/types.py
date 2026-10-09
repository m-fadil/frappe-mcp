from typing import Any, Literal, Union

from pydantic import BaseModel, StrictFloat, StrictInt, StrictStr

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
# MCP-reserved codes (-32020..-32099), protocol 2026-07-28
HEADER_MISMATCH = -32020
UNSUPPORTED_PROTOCOL_VERSION = -32022

# Basic JSON-RPC Types
JSONRPC_VERSION = '2.0'
RequestId = Union[StrictStr, StrictInt, StrictFloat, None]


def dump(model: BaseModel) -> dict[str, Any]:
    """Serialize a model into an MCP result dict (unset optional fields omitted)."""
    return model.model_dump(exclude_none=True, by_alias=True)


class JSONRPCRequest(BaseModel):
    jsonrpc: Literal['2.0']
    # Strict: `true` must not be coerced to 1, or the response id would differ
    # from the request id.
    id: StrictStr | StrictInt | StrictFloat
    method: StrictStr
    params: dict[str, Any] | None = None


class Error(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JSONRPCErrorResponse(BaseModel):
    jsonrpc: str = JSONRPC_VERSION
    id: RequestId
    error: Error


class JSONRPCSuccessResponse(BaseModel):
    jsonrpc: str = JSONRPC_VERSION
    id: RequestId
    result: dict[str, Any]


# General MCP Types from schema.ts


class BaseMetadata(BaseModel):
    name: str
    title: str | None = None


# Content blocks


class TextResourceContents(BaseModel):
    uri: str
    mimeType: str | None = None
    text: str


class BlobResourceContents(BaseModel):
    uri: str
    mimeType: str | None = None
    blob: str  # base64 encoded


class Resource(BaseMetadata):
    uri: str
    description: str | None = None
    mimeType: str | None = None
    annotations: dict[str, Any] | None = None
    size: int | None = None


class TextContent(BaseModel):
    type: str = 'text'
    text: str
    annotations: dict[str, Any] | None = None


class ImageContent(BaseModel):
    type: str = 'image'
    data: str  # base64
    mimeType: str
    annotations: dict[str, Any] | None = None


class AudioContent(BaseModel):
    type: str = 'audio'
    data: str  # base64
    mimeType: str
    annotations: dict[str, Any] | None = None


class ResourceLink(Resource):
    type: str = 'resource_link'


class EmbeddedResource(BaseModel):
    type: str = 'resource'
    resource: TextResourceContents | BlobResourceContents
    annotations: dict[str, Any] | None = None


ContentBlock = Union[
    TextContent, ImageContent, AudioContent, ResourceLink, EmbeddedResource
]


# prompts/get
class GetPromptRequestParams(BaseModel):
    name: str
    arguments: dict[str, str] | None = None


class PromptMessage(BaseModel):
    role: str
    content: ContentBlock


class GetPromptResult(BaseModel):
    description: str | None = None
    messages: list[PromptMessage]


# prompts/list
class ListPromptsRequestParams(BaseModel):
    cursor: str | None = None


class PromptArgument(BaseMetadata):
    description: str | None = None
    required: bool | None = None


class Prompt(BaseMetadata):
    description: str | None = None
    arguments: list[PromptArgument] | None = None


class ListPromptsResult(BaseModel):
    prompts: list[Prompt]
    nextCursor: str | None = None


# tools/call
class CallToolRequestParams(BaseModel):
    name: str
    arguments: dict[str, Any] | None = None


class CallToolResult(BaseModel):
    content: list[ContentBlock]
    structuredContent: dict[str, Any] | None = None
    isError: bool | None = None


# tools/list
class ListToolsRequestParams(BaseModel):
    cursor: str | None = None


class ToolAnnotations(BaseModel):
    title: str | None = None
    readOnlyHint: bool | None = None
    destructiveHint: bool | None = None
    idempotentHint: bool | None = None
    openWorldHint: bool | None = None


class Tool(BaseMetadata):
    description: str | None = None
    inputSchema: dict[str, Any]
    outputSchema: dict[str, Any] | None = None
    annotations: ToolAnnotations | None = None


class ListToolsResult(BaseModel):
    tools: list[Tool]
    nextCursor: str | None = None
