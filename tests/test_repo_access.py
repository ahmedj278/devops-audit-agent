import io
import subprocess
import tarfile
import urllib.error

import pytest

import repo_access
from repo_access import (InvalidInputError, RepoAccessError, RepoRef, _safe_extract, clean_env, clone_repo,
                         github_api_precheck, parse_github_url)


@pytest.mark.parametrize("url,expected", [
    ("https://github.com/octocat/Hello-World", ("octocat", "Hello-World")),
    ("https://github.com/octocat/Hello-World/", ("octocat", "Hello-World")),
    ("https://github.com/octocat/Hello-World.git", ("octocat", "Hello-World")),
    ("https://www.github.com/a/b.c_d-e", ("a", "b.c_d-e")),
    ("  https://github.com/a/b  ", ("a", "b")),
])
def test_parse_valid(url, expected):
    ref = parse_github_url(url)
    assert (ref.owner, ref.name) == expected and ref.url == f"https://github.com/{expected[0]}/{expected[1]}"


@pytest.mark.parametrize("url", [
    "", "   ", "not a url", "http://github.com/a/b", "https://gitlab.com/a/b", "https://github.com/a",
    "https://github.com/a/b/tree/main", "git@github.com:a/b.git", "https://github.com/a/b;rm -rf",
    "https://github.com/../b", "file:///etc/passwd", "https://evil.com/github.com/a/b", None,
])
def test_parse_invalid(url):
    with pytest.raises(InvalidInputError):
        parse_github_url(url)


def test_clean_env_never_passes_secrets(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(k, "super-secret-value")
    env = clean_env()
    assert not any("super-secret" in v for v in env.values())
    assert env["GIT_TERMINAL_PROMPT"] == "0"


def _tar_with(members):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "sym":
                info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
                tar.addfile(info)
            else:
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
    buf.seek(0)
    return tarfile.open(fileobj=buf, mode="r:gz")


def test_safe_extract_strips_top_dir_and_skips_symlinks(tmp_path):
    tar = _tar_with([("o-r-sha/README.md", b"hi", "file"), ("o-r-sha/link", b"", "sym")])
    _safe_extract(tar, tmp_path)
    assert (tmp_path / "README.md").read_text() == "hi" and not (tmp_path / "link").exists()


def test_safe_extract_blocks_path_traversal(tmp_path):
    tar = _tar_with([("o-r-sha/../../evil.txt", b"x", "file")])
    with pytest.raises(RepoAccessError, match="unsafe"):
        _safe_extract(tar, tmp_path / "dest")
    assert not (tmp_path / "evil.txt").exists()


def _http_error(code):
    return urllib.error.HTTPError("u", code, "m", {}, None)


def test_precheck_404_is_definitive(monkeypatch):
    monkeypatch.setattr(repo_access.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(_http_error(404)))
    with pytest.raises(RepoAccessError, match="not found"):
        github_api_precheck(RepoRef("a", "b"))


@pytest.mark.parametrize("exc", [_http_error(403), _http_error(429), urllib.error.URLError("dns"), TimeoutError()])
def test_precheck_soft_failures_fall_through(monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(repo_access.urllib.request, "urlopen", boom)
    assert github_api_precheck(RepoRef("a", "b")) is None


def test_precheck_rejects_private_and_huge(monkeypatch):
    import json

    def fake(payload):
        class R:
            def __enter__(s): return s
            def __exit__(s, *a): return False
            def read(s): return json.dumps(payload).encode()
        return lambda *a, **k: R()

    monkeypatch.setattr(repo_access.urllib.request, "urlopen", fake({"private": True}))
    with pytest.raises(RepoAccessError, match="private"):
        github_api_precheck(RepoRef("a", "b"))
    monkeypatch.setattr(repo_access.urllib.request, "urlopen", fake({"private": False, "size": 10**9}))
    with pytest.raises(RepoAccessError, match="too large"):
        github_api_precheck(RepoRef("a", "b"))
    monkeypatch.setattr(repo_access.urllib.request, "urlopen", fake({"private": False, "size": 100}))
    assert github_api_precheck(RepoRef("a", "b"))["size"] == 100


def test_clone_error_mapping(monkeypatch, tmp_path):
    def fake_run(cmd, timeout, cwd=None):
        return subprocess.CompletedProcess(cmd, 128, "", "fatal: repository 'x' not found")
    monkeypatch.setattr(repo_access, "_run_group", fake_run)
    with pytest.raises(RepoAccessError, match="not found"):
        clone_repo(RepoRef("a", "b"), tmp_path / "d")

    def timeout_run(cmd, timeout, cwd=None):
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(repo_access, "_run_group", timeout_run)
    with pytest.raises(RepoAccessError, match="timed out"):
        clone_repo(RepoRef("a", "b"), tmp_path / "d", timeout=1)


def test_clone_command_is_hardened(monkeypatch, tmp_path):
    captured = {}

    def fake_run(cmd, timeout, cwd=None):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(repo_access, "_run_group", fake_run)
    clone_repo(RepoRef("a", "b"), tmp_path / "d")
    joined = " ".join(captured["cmd"])
    for needle in ("core.hooksPath=/dev/null", "core.symlinks=false", "--depth", "--no-recurse-submodules", "protocol.file.allow=never"):
        assert needle in joined


def test_real_group_timeout_kills_process():
    with pytest.raises(subprocess.TimeoutExpired):
        repo_access._run_group(["sleep", "30"], timeout=0.5)
