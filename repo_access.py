"""Safe access to a public GitHub repository.

* strict URL validation (https://github.com/<owner>/<repo> only)
* optional GitHub API precheck (existence, size, rate limits)
* hardened shallow ``git clone`` (no hooks, no symlinks, no submodules, no prompts)
* tarball fallback with path-traversal protection when ``git`` is unavailable

Nothing in this module ever executes code from the target repository.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("devops_agent.repo")

MAX_REPO_SIZE_KB = 750_000  # refuse repositories larger than ~750 MB
_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


class InvalidInputError(Exception):
    """The repository URL is malformed or unsupported."""


class RepoAccessError(Exception):
    """The repository could not be fetched (not found, timeout, too large, ...)."""


@dataclass(frozen=True)
class RepoRef:
    owner: str
    name: str

    @property
    def url(self) -> str:
        return f"https://github.com/{self.owner}/{self.name}"

    @property
    def clone_url(self) -> str:
        return self.url + ".git"


def parse_github_url(url: str) -> RepoRef:
    if not isinstance(url, str) or not url.strip():
        raise InvalidInputError("repository URL is empty")
    u = url.strip()
    m = re.fullmatch(r"https://(?:www\.)?github\.com/([^/\s]+)/([^/\s]+?)(?:\.git)?/?", u, re.I)
    if not m:
        raise InvalidInputError(
            f"unsupported repository URL {url!r}; expected https://github.com/<owner>/<repo>"
        )
    owner, name = m.group(1), m.group(2)
    if not _NAME_RE.match(owner) or not _NAME_RE.match(name) or name in {".", ".."} or owner in {".", ".."}:
        raise InvalidInputError(f"invalid owner/repository name in {url!r}")
    return RepoRef(owner, name)


def _clean_env() -> dict[str, str]:
    """Environment for subprocesses: pass proxy/cert vars, never API keys or tokens."""
    keep = {
        "PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "TMP", "TEMP", "SYSTEMROOT",
        "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
        "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "GIT_SSL_CAINFO", "CURL_CA_BUNDLE",
    }
    env = {k: v for k, v in os.environ.items() if k in keep}
    env.update({"GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1", "GIT_ASKPASS": "true"})
    return env


def clean_env() -> dict[str, str]:
    return _clean_env()


def github_api_precheck(ref: RepoRef, token: str | None = None, timeout: float = 15.0) -> dict | None:
    """Query the GitHub API for basic metadata.

    Returns the metadata dict, or ``None`` if the API could not be used (rate limit,
    network problem) - the caller then simply tries the clone.
    Raises RepoAccessError for definitive failures (404, too large).
    """
    req = urllib.request.Request(
        f"https://api.github.com/repos/{ref.owner}/{ref.name}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "devops-audit-agent"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (fixed https host)
            data = json.loads(resp.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise RepoAccessError(
                f"repository {ref.url} was not found (it may not exist or may be private)"
            ) from None
        log.warning("GitHub API precheck returned HTTP %s; continuing with clone", e.code)
        return None
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as e:
        log.warning("GitHub API precheck failed (%s); continuing with clone", type(e).__name__)
        return None
    if not isinstance(data, dict):
        return None
    if data.get("private") is True:
        raise RepoAccessError(f"repository {ref.url} is private; only public repositories are supported")
    size = data.get("size")
    if isinstance(size, int) and size > MAX_REPO_SIZE_KB:
        raise RepoAccessError(f"repository is too large ({size // 1024} MB) to audit safely")
    return data


def _run_group(cmd: list[str], timeout: float, cwd: str | None = None) -> subprocess.CompletedProcess:
    """Run a command in its own process group so a timeout kills the whole tree."""
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=_clean_env(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, errors="replace", start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.communicate()
        raise
    return subprocess.CompletedProcess(cmd, proc.returncode, out, err)


def clone_repo(ref: RepoRef, dest: Path, depth: int = 200, timeout: float = 180.0) -> None:
    """Shallow, hardened clone into *dest* (must not exist or be empty)."""
    cmd = [
        "git",
        "-c", "core.hooksPath=/dev/null",
        "-c", "core.symlinks=false",
        "-c", "core.fsmonitor=false",
        "-c", "protocol.file.allow=never",
        "-c", "protocol.ext.allow=never",
        "-c", "submodule.recurse=false",
        "clone", "--depth", str(depth), "--single-branch", "--no-tags",
        "--no-recurse-submodules", ref.clone_url, str(dest),
    ]
    try:
        res = _run_group(cmd, timeout)
    except subprocess.TimeoutExpired:
        raise RepoAccessError(f"timed out after {int(timeout)}s while cloning {ref.url}") from None
    if res.returncode != 0:
        err = (res.stderr or "").strip().splitlines()
        tail = " | ".join(err[-3:]) if err else "unknown error"
        low = tail.lower()
        if "not found" in low or "could not read username" in low or "authentication" in low:
            raise RepoAccessError(f"repository {ref.url} was not found or is not public")
        raise RepoAccessError(f"git clone failed: {tail[:300]}")


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    dest = dest.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk() or member.isdev():
            continue  # links/devices are skipped, never created
        parts = Path(member.name).parts[1:]  # drop the top-level "<owner>-<repo>-<sha>/"
        if not parts:
            continue
        target = (dest.joinpath(*parts)).resolve()
        if dest != target and dest not in target.parents:
            raise RepoAccessError(f"unsafe path in archive: {member.name!r}")
        if member.isdir():
            target.mkdir(parents=True, exist_ok=True)
        elif member.isfile():
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with open(target, "wb") as fh:
                shutil.copyfileobj(src, fh)


def download_tarball(ref: RepoRef, dest: Path, token: str | None = None, timeout: float = 120.0) -> None:
    """Fallback when git is unavailable: fetch the default-branch tarball (no history)."""
    req = urllib.request.Request(
        f"https://api.github.com/repos/{ref.owner}/{ref.name}/tarball",
        headers={"User-Agent": "devops-audit-agent"},
    )
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as fh:  # noqa: S310
                shutil.copyfileobj(resp, fh)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise RepoAccessError(f"repository {ref.url} was not found or is not public") from None
            raise RepoAccessError(f"GitHub returned HTTP {e.code} while downloading {ref.url}") from None
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise RepoAccessError(f"network error while downloading {ref.url}: {type(e).__name__}") from None
        try:
            with tarfile.open(tmp_path, "r:gz") as tar:
                _safe_extract(tar, dest)
        except tarfile.TarError as e:
            raise RepoAccessError(f"downloaded archive is corrupt: {e}") from None
    finally:
        tmp_path.unlink(missing_ok=True)


def fetch_repository(ref: RepoRef, dest: Path, token: str | None = None, timeout: float = 180.0) -> bool:
    """Fetch *ref* into *dest*. Returns True if git history is available."""
    github_api_precheck(ref, token)
    if shutil.which("git"):
        clone_repo(ref, dest, timeout=timeout)
        return True
    log.warning("git is not installed; falling back to tarball download (no commit history)")
    download_tarball(ref, dest, token, timeout=timeout)
    return False
