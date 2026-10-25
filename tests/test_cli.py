"""End-to-end CLI tests (network replaced by a local fixture repo and a scripted LLM)."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import agent as cli
import repo_access
from findings import validate_report
from llm_client import LLMUnavailableError
from tests.fakes import FAKE_AWS_KEY, ScriptedLLM, call, reply

ROOT = Path(__file__).resolve().parent.parent
URL = "https://github.com/example/demo-project"


@pytest.fixture()
def no_env(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(ROOT)  # no stray .env


@pytest.fixture()
def local_fetch(monkeypatch, bad_repo):
    """Replace network fetch by copying the fixture repo into the destination."""
    def fetch(ref, dest, token=None, timeout=180.0):
        shutil.copytree(bad_repo, dest)
        return True
    monkeypatch.setattr(repo_access, "fetch_repository", fetch)


def finding_policy(messages, n):
    if n == 1:
        return reply(call("list_files"), call("scan_secrets"))
    if n == 2:
        return reply(call("read_file", path="config.py"))
    if n == 3:
        return reply(call("report_violation", category="security", severity="critical", file="config.py", line=1,
                          description="Hardcoded AWS access key",
                          evidence=f"config.py line 1 assigns a literal AWS key id ({FAKE_AWS_KEY}) to a variable",
                          recommendation="Use a secrets manager or environment variable; rotate the key"))
    return reply(call("finish_audit"))


def test_invalid_url_exit_2(no_env, tmp_path, capsys):
    assert cli.main(["--repo", "https://example.com/x", "--output", str(tmp_path / "r.json")]) == 2
    assert "unsupported repository URL" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_unwritable_output_exit_2(no_env, tmp_path, capsys):
    assert cli.main(["--repo", URL, "--output", str(tmp_path)]) == 2
    assert "cannot write output" in capsys.readouterr().err


def test_missing_api_key_exit_3(no_env, tmp_path, capsys):
    assert cli.main(["--repo", URL, "--output", str(tmp_path / "r.json")]) == 3
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_repo_not_found_exit_4(no_env, monkeypatch, tmp_path, capsys):
    def fetch(*a, **k):
        raise repo_access.RepoAccessError("repository https://github.com/example/demo-project was not found")
    monkeypatch.setattr(repo_access, "fetch_repository", fetch)
    code = cli.main(["--repo", URL, "--output", str(tmp_path / "r.json")], llm=ScriptedLLM(lambda m, n: reply()))
    assert code == 4 and "not found" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()


def test_full_run_writes_valid_redacted_report(no_env, local_fetch, tmp_path, capsys):
    out = tmp_path / "report.json"
    code = cli.main(["--repo", URL + ".git", "--output", str(out)], llm=ScriptedLLM(finding_policy))
    assert code == 0
    text = out.read_text()
    report = json.loads(text)
    assert validate_report(report) == []
    assert report["repository"] == URL
    v = report["violations"][0]
    assert v["rule_id"] == "SEC-001" and v["file"] == "config.py" and v["line"] == 1
    assert FAKE_AWS_KEY not in text and FAKE_AWS_KEY not in capsys.readouterr().err


def test_empty_report_is_valid(no_env, local_fetch, tmp_path):
    def policy(messages, n):
        return reply(call("list_files"), call("find_files", pattern="*"), call("git_log")) if n == 1 else reply(call("finish_audit"))

    out = tmp_path / "r.json"
    assert cli.main(["--repo", URL, "--output", str(out)], llm=ScriptedLLM(policy)) == 0
    assert json.loads(out.read_text()) == {"repository": URL, "violations": []}


def test_llm_failure_exit_3_without_findings_writes_nothing(no_env, local_fetch, tmp_path, capsys):
    llm = ScriptedLLM(lambda m, n: LLMUnavailableError("provider down"))
    out = tmp_path / "r.json"
    assert cli.main(["--repo", URL, "--output", str(out)], llm=llm) == 3
    assert "provider down" in capsys.readouterr().err and not out.exists()


def test_llm_failure_after_findings_writes_partial_report(no_env, local_fetch, tmp_path):
    def policy(messages, n):
        return finding_policy(messages, n) if n <= 3 else LLMUnavailableError("rate limited out")

    out = tmp_path / "r.json"
    assert cli.main(["--repo", URL, "--output", str(out)], llm=ScriptedLLM(policy)) == 3
    assert len(json.loads(out.read_text())["violations"]) == 1


def test_step_budget_still_exits_0_with_valid_report(no_env, local_fetch, tmp_path):
    out = tmp_path / "r.json"
    code = cli.main(["--repo", URL, "--output", str(out), "--max-steps", "2"],
                    llm=ScriptedLLM(lambda m, n: reply(call("list_files"))))
    assert code == 0 and validate_report(json.loads(out.read_text())) == []


def test_unexpected_error_exit_5(no_env, local_fetch, monkeypatch, tmp_path, capsys):
    import agent_loop
    monkeypatch.setattr(agent_loop.AuditAgent, "run", lambda self, url: (_ for _ in ()).throw(RuntimeError("boom")))
    code = cli.main(["--repo", URL, "--output", str(tmp_path / "r.json")], llm=ScriptedLLM(lambda m, n: reply()))
    assert code == 5 and "boom" in capsys.readouterr().err


def test_exact_cli_format_via_subprocess(tmp_path):
    """`python agent.py --repo <url> --output report.json` is the public contract."""
    env = {"PATH": __import__("os").environ["PATH"], "HOME": str(tmp_path)}
    r = subprocess.run([sys.executable, "agent.py", "--repo", "https://example.com/nope", "--output", str(tmp_path / "report.json")],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "error:" in r.stderr
    r = subprocess.run([sys.executable, "agent.py", "--repo", URL, "--output", str(tmp_path / "report.json")],
                       cwd=ROOT, capture_output=True, text=True, env=env)
    assert r.returncode == 3 and "API_KEY" in r.stderr
    r = subprocess.run([sys.executable, "agent.py", "--help"], cwd=ROOT, capture_output=True, text=True, env=env)
    assert "--repo" in r.stdout and "--output" in r.stdout
