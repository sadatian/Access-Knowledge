import pytest

def test_mcp_tool_schema_validation():
    tool = {
        "name": "search_knowledge_base",
        "description": "Search internal documentation",
        "inputSchema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
    assert tool["name"] == "search_knowledge_base"
    assert "query" in tool["inputSchema"]["required"]
