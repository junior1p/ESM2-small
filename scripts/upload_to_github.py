"""
upload_to_github.py
===================
Create a GitHub release and upload assets (e.g. model checkpoints) to it.

Requires a GitHub personal access token with ``repo`` scope, set via the
``GITHUB_TOKEN`` environment variable or the ``--token`` argument.

Usage
-----
# Create a release and upload a checkpoint:
python upload_to_github.py \\
    --repo owner/repo-name \\
    --tag v1.0.0 \\
    --name "Release v1.0.0" \\
    --notes "Initial release." \\
    --assets checkpoint.pt

# Read release notes from a file:
python upload_to_github.py \\
    --repo owner/repo-name \\
    --tag v1.0.0 \\
    --name "Release v1.0.0" \\
    --notes_file RELEASE_NOTES.md \\
    --assets checkpoint.pt model_config.json

# Dry run (no actual API calls):
python upload_to_github.py \\
    --repo owner/repo-name \\
    --tag v1.0.0 \\
    --name "Release v1.0.0" \\
    --assets checkpoint.pt \\
    --dry_run
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------

GITHUB_API_BASE = "https://api.github.com"
GITHUB_UPLOADS_BASE = "https://uploads.github.com"


def _make_request(
    url: str,
    method: str = "GET",
    data: Optional[bytes] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: int = 60,
) -> Dict:
    """Make an HTTP request and return the parsed JSON response.

    Parameters
    ----------
    url : str
        Request URL.
    method : str
        HTTP method (default: ``"GET"``).
    data : bytes, optional
        Request body.
    headers : dict, optional
        Additional HTTP headers.
    timeout : int
        Request timeout in seconds (default: 60).

    Returns
    -------
    dict
        Parsed JSON response body.

    Raises
    ------
    RuntimeError
        If the HTTP response status is not 2xx.
    """
    req = urllib.request.Request(url, data=data, method=method)
    if headers:
        for key, value in headers.items():
            req.add_header(key, value)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {e.code} {e.reason} for {url}\nResponse: {body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error for {url}: {e.reason}") from e


def _auth_headers(token: str) -> Dict[str, str]:
    """Build standard GitHub API auth headers."""
    return {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }


# ---------------------------------------------------------------------------
# Release management
# ---------------------------------------------------------------------------


def get_release_by_tag(repo: str, tag: str, token: str) -> Optional[Dict]:
    """Fetch an existing release by tag name, or return None if not found.

    Parameters
    ----------
    repo : str
        Repository in ``owner/name`` format.
    tag : str
        Git tag name.
    token : str
        GitHub personal access token.

    Returns
    -------
    dict or None
        Release object dict, or ``None`` if not found.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/releases/tags/{tag}"
    try:
        return _make_request(url, headers=_auth_headers(token))
    except RuntimeError as e:
        if "HTTP 404" in str(e):
            return None
        raise


def create_release(
    repo: str,
    tag: str,
    name: str,
    notes: str,
    token: str,
    draft: bool = False,
    prerelease: bool = False,
) -> Dict:
    """Create a new GitHub release.

    Parameters
    ----------
    repo : str
        Repository in ``owner/name`` format.
    tag : str
        Git tag name for the release.
    name : str
        Release title.
    notes : str
        Release body / notes (Markdown supported).
    token : str
        GitHub personal access token.
    draft : bool
        If True, create as a draft release (default: False).
    prerelease : bool
        If True, mark as pre-release (default: False).

    Returns
    -------
    dict
        Created release object.
    """
    url = f"{GITHUB_API_BASE}/repos/{repo}/releases"
    payload = json.dumps({
        "tag_name": tag,
        "name": name,
        "body": notes,
        "draft": draft,
        "prerelease": prerelease,
    }).encode("utf-8")

    print(f"Creating release '{name}' (tag={tag}) on {repo}...")
    release = _make_request(
        url, method="POST", data=payload, headers=_auth_headers(token)
    )
    print(f"  Release created: {release.get('html_url', '(no URL)')}")
    return release


def upload_asset(
    repo: str,
    release_id: int,
    asset_path: str,
    token: str,
    timeout: int = 600,
) -> Dict:
    """Upload a file as a release asset.

    Parameters
    ----------
    repo : str
        Repository in ``owner/name`` format.
    release_id : int
        Numeric release ID (from the release object).
    asset_path : str
        Local path to the file to upload.
    token : str
        GitHub personal access token.
    timeout : int
        Upload timeout in seconds (default: 600).

    Returns
    -------
    dict
        Uploaded asset object.
    """
    filename = Path(asset_path).name
    file_size = os.path.getsize(asset_path)

    # Correct upload URL format
    url = (
        f"{GITHUB_UPLOADS_BASE}/repos/{repo}/releases/{release_id}"
        f"/assets?name={filename}"
    )

    content_type, _ = mimetypes.guess_type(asset_path)
    if content_type is None:
        content_type = "application/octet-stream"

    print(f"  Uploading '{filename}' ({file_size / 1024 / 1024:.2f} MB)...")

    with open(asset_path, "rb") as f:
        data = f.read()

    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": content_type,
        "Content-Length": str(file_size),
    }

    asset = _make_request(url, method="POST", data=data, headers=headers, timeout=timeout)
    print(f"  Asset uploaded: {asset.get('browser_download_url', '(no URL)')}")
    return asset


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Create a GitHub release and upload model checkpoint assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo", type=str, required=True,
        help="GitHub repository in 'owner/name' format.",
    )
    parser.add_argument(
        "--tag", type=str, required=True,
        help="Git tag name for the release (e.g. 'v1.0.0').",
    )
    parser.add_argument(
        "--name", type=str, required=True,
        help="Release title.",
    )
    parser.add_argument(
        "--notes", type=str, default="",
        help="Release notes / body text (Markdown supported).",
    )
    parser.add_argument(
        "--notes_file", type=str, default=None,
        help=(
            "Path to a file containing release notes. "
            "If set, overrides --notes."
        ),
    )
    parser.add_argument(
        "--assets", type=str, nargs="*", default=[],
        help="Paths to files to upload as release assets.",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help=(
            "GitHub personal access token. "
            "If not set, reads from GITHUB_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--draft", action="store_true", default=False,
        help="Create as a draft release.",
    )
    parser.add_argument(
        "--prerelease", action="store_true", default=False,
        help="Mark as a pre-release.",
    )
    parser.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Print what would be done without making any API calls.",
    )
    args = parser.parse_args()

    # Resolve token
    token = args.token or os.environ.get("GITHUB_TOKEN", "")
    if not token and not args.dry_run:
        print(
            "[ERROR] No GitHub token provided. "
            "Set GITHUB_TOKEN environment variable or use --token.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Resolve release notes
    notes = args.notes
    if args.notes_file is not None:
        notes_path = args.notes_file
        if not os.path.isfile(notes_path):
            print(f"[ERROR] Notes file not found: {notes_path}", file=sys.stderr)
            sys.exit(1)
        with open(notes_path) as f:
            notes = f.read()
        print(f"Release notes loaded from: {notes_path}")

    # Validate assets
    for asset_path in args.assets:
        if not os.path.isfile(asset_path):
            print(f"[ERROR] Asset file not found: {asset_path}", file=sys.stderr)
            sys.exit(1)

    if args.dry_run:
        print("=== DRY RUN (no API calls will be made) ===")
        print(f"  Repo       : {args.repo}")
        print(f"  Tag        : {args.tag}")
        print(f"  Name       : {args.name}")
        print(f"  Draft      : {args.draft}")
        print(f"  Pre-release: {args.prerelease}")
        print(f"  Notes      : {notes[:200]}{'...' if len(notes) > 200 else ''}")
        print(f"  Assets     : {args.assets}")
        return

    # Check if release already exists
    existing = get_release_by_tag(args.repo, args.tag, token)
    if existing is not None:
        print(
            f"[WARN] Release with tag '{args.tag}' already exists: "
            f"{existing.get('html_url', '')}"
        )
        release = existing
    else:
        release = create_release(
            repo=args.repo,
            tag=args.tag,
            name=args.name,
            notes=notes,
            token=token,
            draft=args.draft,
            prerelease=args.prerelease,
        )

    release_id = release["id"]

    # Upload assets
    if args.assets:
        print(f"\nUploading {len(args.assets)} asset(s) to release {release_id}...")
        for asset_path in args.assets:
            upload_asset(
                repo=args.repo,
                release_id=release_id,
                asset_path=asset_path,
                token=token,
            )

    print(f"\n=== Done ===")
    print(f"Release URL: {release.get('html_url', '(unknown)')}")


if __name__ == "__main__":
    main()
