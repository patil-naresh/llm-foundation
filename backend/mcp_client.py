
import json
import logging
import os
import sys

from pathlib import Path
from contextlib import asynccontextmanager

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MCP_SERVER_PATH = PROJECT_ROOT / "mcp" / "server.py"


@asynccontextmanager
async def create_mcp_client():
    """
    Start the MCP server as a subprocess and establish
    an MCP client session.
    """

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_PATH)],
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
    )

    async with stdio_client(server_params) as (read, write):

        async with ClientSession(read, write) as session:

            await session.initialize()

            logger.info("Connected to MCP server")

            yield session

            logger.info("MCP client session closed")


def convert_mcp_tools_to_openai(mcp_tools):
    """
    Convert MCP tool definitions into OpenAI function tools.
    """

    openai_tools = []
    tool_mapping = {}

    for tool in mcp_tools:

        # Prefix to avoid collisions with existing local tools.
        openai_name = f"mcp__{tool.name}"

        openai_tools.append({
            "type": "function",
            "function": {
                "name": openai_name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            }
        })

        # OpenAI name -> original MCP tool name
        tool_mapping[openai_name] = tool.name

    return openai_tools, tool_mapping


async def execute_mcp_tool(session, tool_name: str, arguments: dict):
    """
    Execute an MCP tool and convert its result into
    content suitable for an OpenAI tool message.
    """

    result = await session.call_tool(
        tool_name,
        arguments=arguments,
    )

    if result.isError:
        error_text = "\n".join(
            block.text
            for block in result.content
            if hasattr(block, "text")
        )

        return {
            "success": False,
            "error": error_text or "MCP tool execution failed"
        }

    if result.structuredContent is not None:
        return result.structuredContent

    return [
        block.text if hasattr(block, "text")
        else block.model_dump(mode="json", by_alias=True)
        for block in result.content
    ]
