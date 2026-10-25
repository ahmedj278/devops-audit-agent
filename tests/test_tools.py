import json
import os
import shutil
import sys

import pytest

from tests.conftest import make_repo
from tests.fakes import FAKE_AWS_KEY
from tools import RepoTools, summarize_workflow, tool_definitions


@pytest.fixture()
def tools(bad_repo):
    return RepoTools(bad_repo)


# ------------------------------------------------------------------ list_files / find_files
def test_list_files_shows_structure_and_hides_git(tools):
    r = tools.execute("list_files", {})
    assert not r.is_error
    assert "config.py" in r.content and ".github/" in r.content or ".github/workflows/ci.yml" in r.content
    assert ".git/" not in r.content.replace(".github/", "")
    assert tools.listed_files


def test_list_files_collapses_large_directories(tmp_path):
    files = {f"pkg/mod{i}/f{j}.py": "x=1\n" for i in range(30) for j in range(5)}
    files["README.md"] = "hi"
    t = RepoTools(make_repo(tmp_path / "big", files))
    out = t.execute("list_files", {"max_entries": 20}).content
    assert "pkg/" in out and "files)" in out and "expand" in out
    expanded = t.execute("list_files", {"path": "pkg/mod1/"}).content
    assert "f0.py" in expanded


def test_find_files_glob(tools):
    out = tools.execute("find_files", {"pattern": "*.yml"}).content
    assert ".github/workflows/ci.yml" in out
    assert "No files match" in tools.execute("find_files", {"pattern": "README*"}).content


# ------------------------------------------------------------------ read_file
def test_read_file_numbers_lines_and_redacts(tools):
    out = tools.execute("read_file", {"path": "config.py"}).content
    assert "    1 |" in out and "AWS_ACCESS_KEY_ID" in out
    assert FAKE_AWS_KEY not in out and "[REDACTED:aws_access_key_id]" in out


def test_read_file_marks_lines_seen(tools):
    assert not tools.line_seen("config.py", 1)
    tools.execute("read_file", {"path": "config.py", "start_line": 2, "end_line": 3})
    assert tools.line_seen("config.py", 2) and tools.line_seen("config.py", 3)
    assert not tools.line_seen("config.py", 1)


@pytest.mark.parametrize("bad", ["../etc/passwd", "/etc/passwd", "a/../../x", ".git/config", ""])
def test_read_file_rejects_unsafe_paths(tools, bad):
    r = tools.execute("read_file", {"path": bad})
    assert r.is_error


def test_read_file_missing_directory_binary(tools):
    assert "not found" in tools.execute("read_file", {"path": "nope.py"}).content
    assert "directory" in tools.execute("read_file", {"path": ".github"}).content
    out = tools.execute("read_file", {"path": "__pycache__/app.cpython-311.pyc"}).content
    assert "binary file" in out


def test_read_file_paging(tmp_path):
    t = RepoTools(make_repo(tmp_path / "r", {"big.txt": "\n".join(f"line {i}" for i in range(1, 1000))}))
    out = t.execute("read_file", {"path": "big.txt"}).content
    assert "line 250" in out and "line 251" not in out and "start_line=251" in out
    assert "line 251" in t.execute("read_file", {"path": "big.txt", "start_line": 251}).content


def test_read_file_hides_multiline_private_key_even_mid_block(tmp_path):
    body = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEF\nAASCBKcwggSjAgEAAoIBAQC7\n-----END PRIVATE KEY-----\nafter = 1\n"
    t = RepoTools(make_repo(tmp_path / "k", {"key.pem.txt": body}))
    out = t.execute("read_file", {"path": "key.pem.txt", "start_line": 3}).content
    assert "AASCBKcw" not in out


def test_symlinks_are_not_followed(tmp_path):
    repo = make_repo(tmp_path / "s", {"a.txt": "hello"})
    secret = tmp_path / "outside.txt"
    secret.write_text("TOP SECRET")
    os.symlink(secret, repo / "link.txt")
    t = RepoTools(repo)
    assert t.execute("read_file", {"path": "link.txt"}).is_error
    assert "link.txt" not in t.execute("list_files", {}).content


# ------------------------------------------------------------------ search_code
def test_search_code_finds_and_redacts(tools):
    out = tools.execute("search_code", {"pattern": "aws_access"}).content
    assert "config.py:1:" in out and FAKE_AWS_KEY not in out
    assert tools.line_seen("config.py", 1)


def test_search_code_glob_and_no_match(tools):
    assert "No matches" in tools.execute("search_code", {"pattern": "zzz_nothing", "path_glob": "*.py"}).content
    out = tools.execute("search_code", {"pattern": "pytest", "path_glob": "*.yml"}).content
    assert "ci.yml" in out


def test_search_code_invalid_regex_falls_back_to_literal(tools):
    out = tools.execute("search_code", {"pattern": "main("}).content
    assert "searched as literal" in out and "app.py" in out


def test_search_code_rejects_catastrophic_pattern(tools):
    assert tools.execute("search_code", {"pattern": "(a+)+$"}).is_error


def test_search_skips_binary(tools):
    assert "pyc" not in tools.execute("search_code", {"pattern": "binary"}).content


# ------------------------------------------------------------------ git_log
def test_git_log(tools):
    out = tools.execute("git_log", {"max_count": 5}).content
    assert "initial commit" in out and "fix things" in out and "3 reachable" in out
    detailed = tools.execute("git_log", {"include_files": True, "max_count": 1}).content
    assert "change2.txt" in detailed


def test_git_log_without_git(tmp_path):
    (tmp_path / "x.txt").write_text("1")
    assert "not available" in RepoTools(tmp_path).execute("git_log", {}).content


# ------------------------------------------------------------------ workflows
def test_inspect_workflow_surfaces_risk_facts(tools):
    out = tools.execute("inspect_workflow", {"path": ".github/workflows/ci.yml"}).content
    assert "pull_request_target" in out
    assert "checks out untrusted PR head" in out
    assert "attacker-controllable" in out
    assert "branch/mutable ref" in out
    assert "swallows failures" in out
    assert tools.seen_lines[".github/workflows/ci.yml"]  # observations carry verified line numbers


def test_inspect_workflow_good(good_repo):
    t = RepoTools(good_repo)
    data = json.loads(t.execute("inspect_workflow", {"path": ".github/workflows/ci.yml"}).content.split("\n", 1)[1])
    assert data["test_commands_found"] and data["lint_commands_found"]
    assert data["top_level_permissions"] == {"contents": "read"}
    assert not any("pull_request_target" in o for o in data["observations"])


def test_workflow_on_key_parsed_as_boolean_still_works():
    summary, _ = summarize_workflow("on: [push]\njobs:\n  a:\n    runs-on: x\n    steps:\n      - run: echo hi\n")
    assert "push" in summary["triggers"]


def test_workflow_invalid_yaml_reported():
    summary, _ = summarize_workflow("jobs: [unclosed\n  x: : :")
    assert summary["parse_ok"] is False


def test_workflow_without_ci_triggers_noted():
    summary, _ = summarize_workflow("on:\n  schedule:\n    - cron: '0 0 * * *'\njobs:\n  a:\n    runs-on: x\n    steps:\n      - run: pytest\n")
    assert any("does not run on push" in o for o in summary["observations"])


# ------------------------------------------------------------------ scan_secrets
def test_scan_secrets_reports_kind_never_value(tools):
    out = tools.execute("scan_secrets", {}).content
    assert "config.py:1" in out and "aws_access_key_id" in out
    assert FAKE_AWS_KEY not in out
    assert tools.line_seen("config.py", 1)


def test_scan_secrets_tags_test_code_and_ignores_placeholders(tmp_path):
    t = RepoTools(make_repo(tmp_path / "r", {
        "tests/test_x.py": 'password = "hunter2hunter2"\n',
        ".env.example": "API_KEY=your_api_key_here\nTOKEN=${TOKEN}\n",
    }))
    out = t.execute("scan_secrets", {}).content
    assert "tests/test_x.py:1" in out and "test-code" in out
    assert ".env.example" not in out


def test_scan_secrets_clean(good_repo):
    assert "No probable" in RepoTools(good_repo).execute("scan_secrets", {}).content


# ------------------------------------------------------------------ robustness
def test_unknown_tool_and_bad_args(tools):
    assert "unknown tool" in tools.execute("rm_rf", {}).content
    assert tools.execute("read_file", {"bogus": 1}).is_error
    assert tools.execute("read_file", "not-a-dict").is_error
    assert tools.execute("read_file", {"path": "config.py", "start_line": "abc"}).is_error


def test_output_is_capped(tmp_path):
    t = RepoTools(make_repo(tmp_path / "c", {"f.txt": ("x" * 300 + "\n") * 500}), max_output_chars=2000)
    out = t.execute("read_file", {"path": "f.txt"}).content
    assert len(out) <= 2100


def test_tools_never_modify_repo(bad_repo):
    def snapshot():
        return sorted((p.relative_to(bad_repo).as_posix(), p.stat().st_mtime_ns, p.stat().st_size)
                      for p in bad_repo.rglob("*") if p.is_file() and ".git" not in p.parts)

    before = snapshot()
    t = RepoTools(bad_repo)
    for name, args in [("list_files", {}), ("read_file", {"path": "config.py"}), ("search_code", {"pattern": "a"}),
                       ("git_log", {}), ("scan_secrets", {}), ("inspect_workflow", {"path": ".github/workflows/ci.yml"}),
                       ("run_ruff", {})]:
        t.execute(name, args)
    assert snapshot() == before


@pytest.mark.skipif(not (shutil.which("ruff") or os.path.exists(os.path.join(os.path.dirname(sys.executable), "ruff"))),
                    reason="ruff not installed")
def test_run_ruff_flags_unused_import(tools):
    out = tools.execute("run_ruff", {}).content
    assert "F401" in out and "app.py" in out


def test_run_ruff_no_python(tmp_path):
    t = RepoTools(make_repo(tmp_path / "n", {"README.md": "x"}))
    assert "No Python files" in t.execute("run_ruff", {}).content


def test_required_tools_present_and_schemas_valid():
    names = {d["name"] for d in tool_definitions()}
    assert {"list_files", "read_file", "search_code", "git_log"} <= names
    for d in tool_definitions():
        assert d["input_schema"]["type"] == "object" and d["description"]
