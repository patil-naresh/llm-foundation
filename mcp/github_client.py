import os
import json
import httpx

GITHUB_API = "https://api.github.com"


def github_request(method: str, path: str, params=None, json_body=None):
    """
    Make a request to the GitHub API.
    
    This function now supports discovering accessible repositories
    and fetching commits from any accessible repository, not just
    those owned by the authenticated user.
    """
    token = os.getenv("GITHUB_TOKEN")

    if not token:
        raise RuntimeError("GITHUB_TOKEN is not configured")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2026-03-10",
    }

    url = f"{GITHUB_API}{path}"

    try:
        with httpx.Client(timeout=20.0) as client:
            response = client.request(
                method,
                url,
                headers=headers,
                params=params,
                json=json_body,
            )

            response.raise_for_status()

            if response.status_code == 204:
                return {"success": True}

            return response.json()

    except httpx.HTTPStatusError as exc:
        raise RuntimeError(
            f"GitHub API error: HTTP {exc.response.status_code}"
        ) from exc

    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Unable to connect to GitHub: {exc}"
        ) from exc


def get_accessible_repos():
    """
    Get all repositories accessible by the authenticated user.
    
    Uses affiliation=owner,collaborator,organization_member to discover:
    - Repositories owned by the user
    - Repositories where the user is a collaborator
    - Repositories in organizations where the user is a member
    
    Returns a dict mapping repo_full_name -> repo_info
    """
    url = "/user/repos?affiliation=owner,collaborator,organization_member"
    repos = {}

    try:
        data = github_request("GET", url)
        if isinstance(data, list):
            for repo in data:
                full_name = repo.get("full_name", "")
                if full_name:
                    repos[full_name] = {
                        "id": repo.get("id"),
                        "name": repo.get("name"),
                        "owner": repo.get("owner", {}).get("login", ""),
                        "private": repo.get("private", False),
                        "html_url": repo.get("html_url"),
                    }
    except RuntimeError:
        # If we can't fetch repos, return empty dict
        pass

    return repos


def get_github_commits(owner: str, repo: str, per_page: int = 10) -> str:
    """
    Fetch the latest commits from a GitHub repository.
    
    This function first discovers all repositories accessible by the
    authenticated user, then uses the repository's actual full_name
    to fetch commits. This allows fetching commits from repositories
    owned by other users or organizations, provided the token has
    access.
    
    The workflow:
    1. Discover all accessible repositories via /user/repos?affiliation=owner,collaborator,organization_member
    2. Find the repository matching the given name
    3. Use the actual owner from the repository's full_name
    4. Fetch commits from GET /repos/{actual_owner}/{repo}/commits
    
    Args:
        owner: The expected owner of the repository (may differ from actual owner)
        repo: The name of the repository
        per_page: The number of commits to fetch
    
    Returns:
        A string containing the fetched commits, or an error message
    """
    logger = __import__("logging").getLogger("github-mcp")
    logger.info("GitHub commits tool invoked: owner=%s, repo=%s", owner, repo)

    # Step 1: Discover all accessible repositories
    accessible = get_accessible_repos()

    # Step 2: Find the repository in the accessible list
    # The user may pass "owner/repo" or just "repo" - we need to match
    repo_key = f"{owner}/{repo}" if owner else repo
    actual_owner = owner  # default to what was passed

    logger.info("Accessible repositories: %s", list(accessible.keys()))
    if repo_key in accessible:
        # Repository found in accessible list, use its actual owner
        actual_owner = accessible[repo_key]["owner"]
    else:
        # Try matching just by repo name (in case owner was different)
        logger.info("Repository %s not found in accessible list, trying to match by name", repo)
        logger.info("accessible repositories: %s", list(accessible.items()))
        for full_name, info in accessible.items():
            if info["name"].lower() == repo.lower():
                repo_key = full_name
                actual_owner = info["owner"]
                break

    # Step 3: Fetch commits using the actual owner
    try:
        commits = github_request(
            "GET",
            f"/repos/{actual_owner}/{repo}/commits",
            params={"per_page": per_page},
        )
        logger.info("GitHub commits fetched: owner=%s, repo=%s, count=%s", actual_owner, repo, len(commits) if isinstance(commits, list) else 0,)
        commit_list = [{
                    "sha": item["sha"][:7],
                    "message": item["commit"]["message"],
                    "author": (
                        item["commit"]["author"] or {}
                    ).get("name"),
                    "date": (
                        item["commit"]["author"] or {}
                    ).get("date"),
                    "url": item["html_url"],
                }
                for item in commits]
        logger.info("Formatted commit list: %s", commit_list)
        return json.dumps(commit_list, indent=2)
    except RuntimeError as e:
        return f"Error fetching GitHub commits: {e}"
