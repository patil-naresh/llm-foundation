from mcp.server.fastmcp import FastMCP
from httpx import HTTPError
import asyncio
from backend_client import BackendClient
import json
from github_client import get_github_commits as github_commits_func
import logging
from pathlib import Path

LOG_FILE = Path(__file__).parent / "mcp_debug.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    force=True
)

logger = logging.getLogger("github-mcp")

mcp = FastMCP("LLM Fundamentals MCP Server")

backend = BackendClient()


@mcp.tool()
async def test_post_request() -> str:
    """
    Test the POST request to the Flask LLM backend.

    Returns a string indicating the success or failure of the request.
    """
    try:
        response = await backend.post(
            "/v1/dummy",
            payload={"message": "Hello, backend!"}
        )
        return f"POST request successful: {response}"
    except HTTPError as e:
        return f"Error during POST request: {e}"

@mcp.tool()
async def get_github_commits(owner: str, repo: str, per_page: int = 10) -> str:
    """
    Fetch the latest commits from a GitHub repository.

    Use this tool whenever the user asks about:
    - Latest GitHub commits
    - Recent repository changes
    - Commit history
    - Who changed the code
    - Repository activity

    Args:
        owner: The owner of the repository.
        repo: The name of the repository.
        per_page: The number of commits to fetch.

    Returns:
        A string containing the fetched commits.
    """
    logger.info("GitHub tool invoked")
    logger.info("GitHub tool called: owner=%s, repo=%s, per_page=%s", owner, repo, per_page)
    try:
        commits = await asyncio.to_thread(github_commits_func, owner, repo, per_page)
        logger.info("GitHub commits fetched successfully")
        return commits
    except RuntimeError as e:
        return f"Error fetching GitHub commits: {e}"


@mcp.tool()
async def get_backend_health() -> str:
    """
    Check whether the Flask LLM backend is running.

    Returns the health status of the backend.
    """
    try:
        health = await backend.check_health()
        return str(health)
    except HTTPError as e:
        return f"Error checking backend health: {e}"

@mcp.resource(
    "llm://backend/status",
    mime_type="application/json"
)
async def get_backend_status() -> str:
    """
    Get the status of the Flask LLM backend.

    Returns a JSON string containing the backend status.
    """
    try:
        health = await backend.check_health()
        return json.dumps({
                "service": "LLM Backend",
                "status": "available",
                "backend_response": json.dumps(health, indent=2)
            }, indent=2)
    except HTTPError as e:
        return json.dumps({"status": "unhealthy", "error": str(e)})


@mcp.resource(
    "llm://github/{owner}/{repo}/commits/{per_page}",
    mime_type="application/json"
)
async def get_github_commits_resource(owner: str, repo: str, per_page: int = 10) -> str:
    """
    Fetch the latest commits from a GitHub repository.

    Args:
        owner: The owner of the repository.
        repo: The name of the repository.
        per_page: The number of commits to fetch.

    Returns:
        A string containing the fetched commits.
    """
    return await get_github_commits(owner, repo, per_page)

@mcp.resource(
    "llm://backend/dummy",
    mime_type="application/json"
)
async def test_post_request() -> str:
    """
    Test the POST request to the Flask LLM backend.

    Returns a string indicating the success or failure of the request.
    """
    try:
        response = await backend.post(
            "/v1/dummy",
            payload={"message": "Hello, backend Test API!"}
        )
        return f"POST request successful: {response}"
    except HTTPError as e:
        return f"Error during POST request: {e}"    


@mcp.prompt()
def explain_code(
    code: str,
    language: str = "Python"
) -> str:
    """
    Generate instructions to explain a code snippet.

    Args:
        code: Source code to explain.
        language: Programming language.
    """

    return f"""
You are an experienced software engineer and technical instructor.

Explain the following {language} code in detail.

Code:
```{language}
{code}
```
"""


if __name__ == "__main__":
    mcp.run()