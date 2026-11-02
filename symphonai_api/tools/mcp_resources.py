"""Read-only MCP resource catalogs and text contents."""
from __future__ import annotations

import base64
import json

from symphonai_api.mcp import McpError
from symphonai_api.models import ToolResult
from symphonai_api.tools.base import LocalTool
from symphonai_api.tools.metadata import ToolEffect, ToolMetadata


class ListMcpResourcesTool(LocalTool):
    name = 'list_mcp_resources'
    description = 'List MCP resources, optionally on one server. Return at most 100 and a next_cursor for another page.'
    parameters = {'type': 'object', 'properties': {'server': {'type': 'string'}, 'cursor': {'type': 'string'}}, 'additionalProperties': False}

    def __init__(self, clients):
        self.clients = dict(clients)

    def metadata(self, arguments):
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=True, paths=())

    def _execute(self, tool_call, policy, cancel=None):
        arguments = tool_call.arguments
        names = [arguments['server']] if 'server' in arguments else sorted(self.clients)
        if any(name not in self.clients for name in names):
            return ToolResult(tool_call.id, False, error='Unknown resource server')
        try:
            index, cursor, offset = 0, None, 0
            if arguments.get('cursor'):
                state = json.loads(base64.urlsafe_b64decode(arguments['cursor']).decode())
                if state['servers'] != names:
                    raise ValueError()
                index, cursor, offset = state['index'], state['cursor'], state['offset']
                if type(index) is not int or not 0 <= index < len(names) or type(offset) is not int or offset < 0 or cursor is not None and not isinstance(cursor, str):
                    raise ValueError()
            resources, seen = [], set()
            while index < len(names):
                if cancel is not None:
                    cancel.raise_if_cancelled()
                marker = (index, cursor, offset)
                if marker in seen:
                    raise McpError('MCP resource pagination repeated a cursor')
                seen.add(marker)
                server = names[index]
                page = self.clients[server].list_resources(cursor)
                entries = page['resources']
                remaining = 100 - len(resources)
                selected = entries[offset:offset + remaining]
                resources.extend({'server': server, 'uri': item['uri'], 'name': item['name'],
                                  'description': item.get('description', ''), 'mimeType': item.get('mimeType', '')}
                                 for item in selected)
                offset += len(selected)
                if offset >= len(entries):
                    cursor, offset = page.get('nextCursor'), 0
                    if cursor is None:
                        index += 1
                if len(resources) == 100:
                    break
            result = {'resources': resources}
            if index < len(names):
                value = {'servers': names, 'index': index, 'cursor': cursor, 'offset': offset}
                result['next_cursor'] = base64.urlsafe_b64encode(json.dumps(value).encode()).decode()
            return ToolResult(tool_call.id, True, content=json.dumps(result, ensure_ascii=False))
        except (McpError, ValueError, KeyError, TypeError):
            return ToolResult(tool_call.id, False, error='Could not list MCP resources (invalid cursor or server response)')


class ReadMcpResourceTool(LocalTool):
    name = 'read_mcp_resource'
    description = 'Read an MCP resource by server and URI. Return text; describe binary data without inlining it.'
    parameters = {'type': 'object', 'properties': {'server': {'type': 'string'}, 'uri': {'type': 'string'}},
                  'required': ['server', 'uri'], 'additionalProperties': False}

    def __init__(self, clients):
        self.clients = dict(clients)

    def metadata(self, arguments):
        return ToolMetadata(effect=ToolEffect.READ_ONLY, concurrency_safe=True, paths=())

    def _execute(self, tool_call, policy, cancel=None):
        client = self.clients.get(tool_call.arguments['server'])
        if client is None:
            return ToolResult(tool_call.id, False, error='Unknown resource server')
        try:
            result = client.read_resource(tool_call.arguments['uri'])
            parts = []
            for item in result['contents']:
                if not isinstance(item, dict):
                    raise McpError('Invalid resource content')
                if isinstance(item.get('text'), str):
                    parts.append(item['text'])
                else:
                    parts.append(f"[Binary resource: {item.get('uri', tool_call.arguments['uri'])}; MIME type: {item.get('mimeType', 'unknown')}; contents omitted]")
            return ToolResult(tool_call.id, True, content='\n'.join(parts))
        except McpError:
            return ToolResult(tool_call.id, False, error='Could not read MCP resource')
